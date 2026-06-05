#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_trt.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TENSORRT_ROOT,
    GENMO_ROOT,
    find_trtexec,
    import_status,
    write_json,
)


def _torch_status() -> dict:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        info = {
            "ok": True,
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        if cuda_available:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
        return info
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _nvidia_smi_status() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"ok": False, "error": "nvidia-smi not found"}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,cuda_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return {"ok": True, "output": out.strip()}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose GENMO realtime TensorRT environment")
    parser.add_argument("--tensorrt-root", default=str(DEFAULT_TENSORRT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT / "diagnostics.json"))
    parser.add_argument("--ckpt-path", default=str(GENMO_ROOT / "inputs/pretrained/gem_smpl.ckpt"))
    parser.add_argument(
        "--hmr2-ckpt",
        default=str(GENMO_ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt"),
    )
    parser.add_argument(
        "--vitpose-ckpt",
        default=str(GENMO_ROOT / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tensorrt_root = Path(args.tensorrt_root).expanduser().resolve()
    trtexec = find_trtexec(tensorrt_root)
    wheel_dir = tensorrt_root / "python"
    cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "cp_tag": cp_tag,
        },
        "paths": {
            "genmo_root": str(GENMO_ROOT),
            "tensorrt_root": str(tensorrt_root),
            "trtexec": trtexec,
            "ckpt_path_exists": Path(args.ckpt_path).exists(),
            "hmr2_ckpt_exists": Path(args.hmr2_ckpt).exists(),
            "vitpose_ckpt_exists": Path(args.vitpose_ckpt).exists(),
            "matching_tensorrt_wheels": sorted(p.name for p in wheel_dir.glob(f"*{cp_tag}*.whl")),
        },
        "torch": _torch_status(),
        "nvidia_smi": _nvidia_smi_status(),
        "modules": {
            "tensorrt": import_status("tensorrt"),
            "onnx": import_status("onnx"),
            "onnxruntime": import_status("onnxruntime"),
            "cv2": import_status("cv2"),
            "ultralytics": import_status("ultralytics"),
        },
    }

    output = write_json(args.output, report)
    print(f"[diagnose] wrote {output}")
    if not report["torch"].get("cuda_available"):
        print("[diagnose] WARNING: torch.cuda.is_available() is false")
    if trtexec is None:
        print("[diagnose] WARNING: trtexec not found")
    if not report["modules"]["tensorrt"]["ok"]:
        print("[diagnose] WARNING: Python TensorRT not importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
