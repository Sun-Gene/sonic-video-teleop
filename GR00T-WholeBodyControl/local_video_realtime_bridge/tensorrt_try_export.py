#!/usr/bin/env python3
"""Experimental GENMO -> ONNX/TensorRT diagnostic helper.

This is intentionally a best-effort tool. The public GENMO PyTorch demo uses a
dictionary-based predict path and diffusion sampling logic, so a clean TensorRT
export may require upstream model changes. The script still gives a practical
starting point:

1. Check TensorRT Python/trtexec availability.
2. Optionally run a small ONNX smoke export around GEM predict().
3. Optionally call trtexec to build an engine from an existing ONNX file.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def _setup_genmo_imports(genmo_repo: Path) -> None:
    os.chdir(genmo_repo)
    for path in (genmo_repo, genmo_repo / "scripts" / "demo"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _find_trtexec(tensorrt_root: Path | None) -> str | None:
    candidates: list[Path] = []
    if tensorrt_root is not None:
        candidates.extend(
            [
                tensorrt_root / "bin" / "trtexec",
                tensorrt_root / "targets" / "x86_64-linux-gnu" / "bin" / "trtexec",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("trtexec")


def _diagnose(tensorrt_root: Path | None) -> str | None:
    print("[diagnose] Python:", sys.executable)
    try:
        import torch

        print("[diagnose] torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    except Exception as exc:
        print("[diagnose] torch import failed:", repr(exc))

    try:
        import tensorrt as trt

        print("[diagnose] tensorrt python:", trt.__version__)
    except Exception as exc:
        print("[diagnose] tensorrt python not importable:", repr(exc))

    trtexec = _find_trtexec(tensorrt_root)
    print("[diagnose] trtexec:", trtexec or "not found")
    return trtexec


def _prepare_example_data(
    *,
    genmo_repo: Path,
    example_video: Path,
    hmr2_ckpt: Path,
    static_cam: bool,
    output_root: Path,
) -> dict:
    import torch
    from demo_smpl_hpe import assemble_data
    from demo_utils import CocoPoseExtractor, detect_and_track, get_camera_static, get_image_features
    from gem.utils.video_io_utils import get_video_lwh, read_video_np

    output_dir = output_root / example_video.stem
    preprocess_dir = output_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    L, W, H = get_video_lwh(str(example_video))
    bbx_xys = detect_and_track(str(example_video), str(preprocess_dir))
    vitpose_cache = preprocess_dir / "vitpose.pt"
    if vitpose_cache.exists():
        kp2d = torch.load(vitpose_cache, map_location="cpu")
    else:
        frames = read_video_np(str(example_video))
        kp2d = CocoPoseExtractor(device="cuda").extract(frames, bbx_xys, batch_size=32)
        torch.save(kp2d, vitpose_cache)
    f_imgseq, has_img_mask = get_image_features(
        str(example_video),
        bbx_xys,
        str(hmr2_ckpt),
        str(preprocess_dir),
    )
    R_w2c, cam_angvel, cam_tvel, K_fullimg = get_camera_static(L, W, H)
    return assemble_data(
        kp2d=kp2d,
        bbx_xys=bbx_xys,
        K_fullimg=K_fullimg,
        cam_angvel=cam_angvel,
        cam_tvel=cam_tvel,
        R_w2c=R_w2c,
        f_imgseq=f_imgseq,
        has_img_mask=has_img_mask,
        static_cam=static_cam,
    )


def _try_smoke_export(args: argparse.Namespace) -> bool:
    if not args.example_video:
        raise ValueError("--example-video is required for --try-smoke-export")

    import torch

    genmo_repo = Path(args.genmo_repo).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    hmr2_ckpt = Path(args.hmr2_ckpt).expanduser().resolve()
    example_video = Path(args.example_video).expanduser().resolve()
    onnx_out = Path(args.onnx_out).expanduser().resolve()
    output_root = Path(args.work_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    _setup_genmo_imports(genmo_repo)
    from demo_utils import load_model

    data = _prepare_example_data(
        genmo_repo=genmo_repo,
        example_video=example_video,
        hmr2_ckpt=hmr2_ckpt,
        static_cam=args.static_cam,
        output_root=output_root / "preprocess",
    )
    model = load_model(str(ckpt_path))

    class GemPredictSmokeWrapper(torch.nn.Module):
        def __init__(self, wrapped_model, wrapped_data: dict, static_cam: bool) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model
            self.wrapped_data = wrapped_data
            self.static_cam = static_cam

        def forward(self, dummy: torch.Tensor):
            _ = dummy
            pred = self.wrapped_model.predict(self.wrapped_data, static_cam=self.static_cam)
            body = pred.get("body_params_global") or pred["body_params_incam"]
            return body["body_pose"], body["global_orient"], body["transl"]

    wrapper = GemPredictSmokeWrapper(model, data, args.static_cam).cuda().eval()
    dummy = torch.zeros(1, device="cuda")
    print("[export] starting ONNX smoke export")
    print("[export] WARNING: this wrapper closes over one example data dict; success does not mean a deployable engine.")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_out),
        opset_version=args.opset,
        input_names=["dummy"],
        output_names=["body_pose", "global_orient", "transl"],
        do_constant_folding=False,
    )
    print(f"[export] wrote {onnx_out}")
    return True


def _build_engine(trtexec: str, onnx_path: Path, engine_path: Path, fp16: bool) -> None:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if fp16:
        cmd.append("--fp16")
    print("[trtexec]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[trtexec] wrote {engine_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Best-effort GENMO TensorRT export diagnostics")
    parser.add_argument("--genmo-repo", default="/path/to/workspace/GENMO")
    parser.add_argument("--ckpt-path", default="/path/to/workspace/GENMO/inputs/pretrained/gem_smpl.ckpt")
    parser.add_argument(
        "--hmr2-ckpt",
        default="/path/to/workspace/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt",
    )
    parser.add_argument("--tensorrt-root", default="/path/to/workspace/TensorRT-10.13.3.9")
    parser.add_argument("--example-video", default=None)
    parser.add_argument(
        "--work-dir",
        default="/path/to/workspace/GR00T-WholeBodyControl/outputs/realtime_bridge_tensorrt",
    )
    parser.add_argument(
        "--onnx-out",
        default="/path/to/workspace/GR00T-WholeBodyControl/outputs/realtime_bridge_tensorrt/gem_predict_smoke.onnx",
    )
    parser.add_argument(
        "--engine-out",
        default="/path/to/workspace/GR00T-WholeBodyControl/outputs/realtime_bridge_tensorrt/gem_predict_smoke.engine",
    )
    parser.add_argument("--try-smoke-export", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--static-cam", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tensorrt_root = Path(args.tensorrt_root).expanduser().resolve() if args.tensorrt_root else None
    trtexec = _diagnose(tensorrt_root)

    if args.try_smoke_export:
        try:
            _setup_genmo_imports(Path(args.genmo_repo).expanduser().resolve())
            _try_smoke_export(args)
        except Exception:
            diagnostic_path = Path(args.work_dir).expanduser().resolve() / "export_failure.txt"
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[export] failed; diagnostic written to {diagnostic_path}")
            print(traceback.format_exc())
            return 2

    if args.build_engine:
        if trtexec is None:
            print("[trtexec] not found; cannot build engine")
            return 3
        _build_engine(
            trtexec=trtexec,
            onnx_path=Path(args.onnx_out).expanduser().resolve(),
            engine_path=Path(args.engine_out).expanduser().resolve(),
            fp16=args.fp16,
        )

    if not args.try_smoke_export and not args.build_engine:
        print("[done] diagnostics only. Add --try-smoke-export and/or --build-engine to experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
