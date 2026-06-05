#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_trt.common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TENSORRT_ROOT,
    GENMO_ROOT,
    add_genmo_to_syspath,
    build_engine_with_trtexec,
    find_trtexec,
    write_json,
)


def _tree_detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _tree_detach(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree_detach(v) for v in value)
    return value


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _tree_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree_to_device(v, device) for v in value)
    return value


def _tensor_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): _tensor_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_summary(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


class CapturedDenoiserWrapper(torch.nn.Module):
    """Trace one GEM denoiser step with captured non-input condition tensors.

    This is intentionally a low-level export attempt. The public GENMO checkpoint
    does not ship a ready TensorRT graph, so the first step is identifying which
    part of the diffusion denoiser can be made ONNX/TensorRT-compatible.
    """

    def __init__(self, denoiser: torch.nn.Module, captured_args: tuple[Any, ...], captured_kwargs: dict[str, Any]):
        super().__init__()
        self.denoiser = denoiser
        self.captured_args = captured_args
        self.captured_kwargs = captured_kwargs

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        args = list(self.captured_args)
        if len(args) < 2:
            raise RuntimeError("Captured denoiser call did not contain xt and timesteps")
        args[0] = xt
        args[1] = timesteps
        return self.denoiser(*args, **self.captured_kwargs)


def _prepare_demo_data(video: str, hmr2_ckpt: str, output_root: Path, static_cam: bool) -> dict[str, Any]:
    from demo_smpl_hpe import assemble_data
    from demo_utils import CocoPoseExtractor, detect_and_track, get_camera_static, get_image_features
    from gem.utils.video_io_utils import get_video_lwh, read_video_np

    preprocess_dir = output_root / Path(video).stem / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    length, width, height = get_video_lwh(video)
    bbx_xys = detect_and_track(video, str(preprocess_dir))
    vitpose_cache = preprocess_dir / "vitpose.pt"
    if vitpose_cache.exists():
        kp2d = torch.load(vitpose_cache, map_location="cpu")
    else:
        frames = read_video_np(video)
        kp2d = CocoPoseExtractor(device="cuda").extract(frames, bbx_xys, batch_size=32)
        torch.save(kp2d, vitpose_cache)
    f_imgseq, has_img_mask = get_image_features(video, bbx_xys, hmr2_ckpt, str(preprocess_dir))
    r_w2c, cam_angvel, cam_tvel, k_fullimg = get_camera_static(length, width, height)
    return assemble_data(
        kp2d=kp2d,
        bbx_xys=bbx_xys,
        K_fullimg=k_fullimg,
        cam_angvel=cam_angvel,
        cam_tvel=cam_tvel,
        R_w2c=r_w2c,
        f_imgseq=f_imgseq,
        has_img_mask=has_img_mask,
        static_cam=static_cam,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attempt to export GEM's diffusion denoiser step to ONNX/TensorRT by "
            "capturing a real denoiser call from an example video."
        )
    )
    parser.add_argument("--example-video", required=True)
    parser.add_argument("--ckpt-path", default=str(GENMO_ROOT / "inputs/pretrained/gem_smpl.ckpt"))
    parser.add_argument(
        "--hmr2-ckpt",
        default=str(GENMO_ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "gem_export_capture"))
    parser.add_argument("--onnx-out", default=str(DEFAULT_OUTPUT_ROOT / "onnx/gem_denoiser_step.onnx"))
    parser.add_argument("--engine-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/gem_denoiser_step.engine"))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_ROOT / "export_gem_denoiser_report.json"))
    parser.add_argument("--tensorrt-root", default=str(DEFAULT_TENSORRT_ROOT))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--static-cam", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    add_genmo_to_syspath()
    report: dict[str, Any] = {
        "module": "gem_denoiser",
        "ok": False,
        "example_video": args.example_video,
        "purpose": "capture one real denoiser call and test ONNX/TensorRT exportability",
    }

    started = time.time()
    captured: dict[str, Any] = {}
    try:
        from demo_utils import load_model

        data = _prepare_demo_data(
            video=args.example_video,
            hmr2_ckpt=args.hmr2_ckpt,
            output_root=Path(args.output_root).expanduser().resolve(),
            static_cam=args.static_cam,
        )
        model = load_model(args.ckpt_path).cuda().eval()
        denoiser = model.pipeline.denoiser3d.denoiser
        original_forward = denoiser.forward

        def capture_forward(*call_args: Any, **call_kwargs: Any) -> Any:
            if not captured:
                captured["args"] = _tree_detach(call_args)
                captured["kwargs"] = _tree_detach(call_kwargs)
                captured["summary"] = {
                    "args": _tensor_summary(captured["args"]),
                    "kwargs": _tensor_summary(captured["kwargs"]),
                }
            return original_forward(*call_args, **call_kwargs)

        denoiser.forward = capture_forward  # type: ignore[method-assign]
        try:
            with torch.no_grad():
                _ = model.predict(data, static_cam=args.static_cam)
        finally:
            denoiser.forward = original_forward  # type: ignore[method-assign]

        if not captured:
            raise RuntimeError("GEM predict finished without calling pipeline.denoiser3d.denoiser")

        captured_args = _tree_to_device(captured["args"], torch.device("cuda"))
        captured_kwargs = _tree_to_device(captured["kwargs"], torch.device("cuda"))
        xt = captured_args[0]
        timesteps = captured_args[1]
        wrapper = CapturedDenoiserWrapper(denoiser, captured_args, captured_kwargs).cuda().eval()
        onnx_out = Path(args.onnx_out).expanduser().resolve()
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            (xt, timesteps),
            str(onnx_out),
            input_names=["xt", "timesteps"],
            output_names=["denoised"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        report.update(
            {
                "ok": True,
                "onnx_path": str(onnx_out),
                "captured_call": captured["summary"],
                "elapsed_s": round(time.time() - started, 3),
            }
        )
        if args.build_engine:
            engine_report = build_engine_with_trtexec(
                onnx_path=onnx_out,
                engine_path=args.engine_out,
                trtexec=find_trtexec(args.tensorrt_root),
                fp16=args.fp16,
                extra_args=["--noDataTransfers"],
            )
            report["engine"] = engine_report
            report["ok"] = bool(engine_report.get("ok"))
    except Exception:
        report["traceback"] = traceback.format_exc()
        report["captured_call"] = captured.get("summary")
        report["elapsed_s"] = round(time.time() - started, 3)
        report["blocker"] = (
            "GEM denoiser ONNX/TensorRT export failed. This is a real blocker for a "
            "paper-level TensorRT runtime; do not treat the PyTorch demo path as equivalent."
        )
        print(report["traceback"])

    output = write_json(args.report, report)
    print(f"[export_gem_denoiser] wrote {output}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
