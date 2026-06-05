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
from realtime_trt.export_gem_denoiser import _prepare_demo_data, _tensor_summary, _tree_detach, _tree_to_device  # noqa: E402


def _slice_time(value: Any, window_frames: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim >= 2 and value.shape[1] >= window_frames:
            return value[:, :window_frames].detach().clone()
        if value.ndim >= 1 and value.shape[0] >= window_frames and value.shape[0] != 1:
            return value[:window_frames].detach().clone()
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _slice_time(v, window_frames) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_slice_time(v, window_frames) for v in value)
    return value


class ExplicitConditionDenoiserWrapper(torch.nn.Module):
    """Export a GEM denoiser call with runtime condition tensors as inputs.

    The remaining nested GENMO dictionaries are captured from a real demo call
    and time-sliced to a fixed runtime window. This is the next practical step
    after the original captured exporter: conditions are no longer baked into
    the engine, but GENMO-specific non-tensor flags and sample-index metadata
    still come from the checkpoint/config.
    """

    def __init__(self, denoiser: torch.nn.Module, captured_args: tuple[Any, ...], captured_kwargs: dict[str, Any]):
        super().__init__()
        self.denoiser = denoiser
        self.captured_args = captured_args
        self.captured_kwargs = captured_kwargs

    def forward(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        f_cond: torch.Tensor,
        f_uncond: torch.Tensor,
        f_empty: torch.Tensor,
    ) -> torch.Tensor:
        args = list(self.captured_args)
        y = dict(args[2])
        y["f_cond"] = f_cond
        y["f_uncond"] = f_uncond
        y["f_empty"] = f_empty
        y["mask"] = torch.ones(xt.shape[:2], dtype=torch.bool, device=xt.device)
        y["length"] = torch.full((xt.shape[0],), xt.shape[1], dtype=torch.long, device=xt.device)
        args[0] = xt
        args[1] = timesteps
        args[2] = y
        out = self.denoiser(*args, **self.captured_kwargs)
        if isinstance(out, dict):
            for key in ("pred_x_start", "pred_x", "pred_xstart"):
                if key in out:
                    return out[key]
            raise RuntimeError(f"Denoiser output dict has no pred_x_start/pred_x key: {list(out)}")
        return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GEM denoiser with explicit runtime condition inputs")
    parser.add_argument("--example-video", required=True)
    parser.add_argument("--ckpt-path", default=str(GENMO_ROOT / "inputs/pretrained/gem_smpl.ckpt"))
    parser.add_argument(
        "--hmr2-ckpt",
        default=str(GENMO_ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt"),
    )
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "gem_export_explicit"))
    parser.add_argument("--onnx-out", default=str(DEFAULT_OUTPUT_ROOT / "onnx/gem_denoiser_explicit_w32.onnx"))
    parser.add_argument("--engine-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/gem_denoiser_explicit_w32.engine"))
    parser.add_argument("--metadata-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/gem_runtime_metadata.json"))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_ROOT / "export_gem_denoiser_explicit_report.json"))
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
        "module": "gem_denoiser_explicit",
        "ok": False,
        "window_frames": args.window_frames,
        "example_video": args.example_video,
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
            raise RuntimeError("GEM predict finished without calling the denoiser")

        captured_args = _tree_to_device(_slice_time(captured["args"], args.window_frames), torch.device("cuda"))
        captured_kwargs = _tree_to_device(_slice_time(captured["kwargs"], args.window_frames), torch.device("cuda"))

        xt = captured_args[0]
        timesteps = captured_args[1]
        y = captured_args[2]
        f_cond = y["f_cond"]
        f_uncond = y["f_uncond"]
        f_empty = y["f_empty"]
        wrapper = ExplicitConditionDenoiserWrapper(denoiser, captured_args, captured_kwargs).cuda().eval()

        onnx_out = Path(args.onnx_out).expanduser().resolve()
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            (xt, timesteps, f_cond, f_uncond, f_empty),
            str(onnx_out),
            input_names=["xt", "timesteps", "f_cond", "f_uncond", "f_empty"],
            output_names=["pred_x_start"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )

        metadata = {
            "ok": True,
            "engine_kind": "gem_denoiser_explicit",
            "window_frames": args.window_frames,
            "motion_dim": int(xt.shape[-1]),
            "condition_dim": int(f_cond.shape[-1]),
            "input_names": ["xt", "timesteps", "f_cond", "f_uncond", "f_empty"],
            "output_names": ["pred_x_start"],
            "shapes": {
                "xt": list(xt.shape),
                "timesteps": list(timesteps.shape),
                "f_cond": list(f_cond.shape),
                "f_uncond": list(f_uncond.shape),
                "f_empty": list(f_empty.shape),
                "pred_x_start": list(xt.shape),
            },
            "diffusion": {
                "num_timesteps": int(model.pipeline.denoiser3d.test_gen_only_diffusion.num_timesteps),
                "alphas_cumprod": model.pipeline.denoiser3d.test_gen_only_diffusion.alphas_cumprod.tolist(),
                "alphas_cumprod_prev": model.pipeline.denoiser3d.test_gen_only_diffusion.alphas_cumprod_prev.tolist(),
                "sqrt_recip_alphas_cumprod": model.pipeline.denoiser3d.test_gen_only_diffusion.sqrt_recip_alphas_cumprod.tolist(),
                "sqrt_recipm1_alphas_cumprod": model.pipeline.denoiser3d.test_gen_only_diffusion.sqrt_recipm1_alphas_cumprod.tolist(),
                "ddim_eta": float(model.pipeline.denoiser3d.model_cfg.diffusion.ddim_eta),
                "clip_denoised": False,
            },
            "note": (
                "Condition tensors are runtime inputs. GENMO condition assembly and SMPL decode "
                "are still handled by Python until their graphs are separately exported."
            ),
        }
        write_json(args.metadata_out, metadata)

        report.update(
            {
                "ok": True,
                "onnx_path": str(onnx_out),
                "metadata_path": str(Path(args.metadata_out).expanduser().resolve()),
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
        print(report["traceback"])

    output = write_json(args.report, report)
    print(f"[export_gem_denoiser_explicit] wrote {output}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
