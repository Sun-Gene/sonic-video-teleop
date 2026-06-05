#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_trt.common import DEFAULT_TENSORRT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install local TensorRT/ONNX deps into the current Python env. "
            "Falls back to uv pip when python -m pip is unavailable."
        )
    )
    parser.add_argument("--tensorrt-root", default=str(DEFAULT_TENSORRT_ROOT))
    parser.add_argument("--with-onnxruntime-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _python_has_pip(python: str) -> bool:
    completed = subprocess.run(
        [python, "-m", "pip", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0


def _build_install_command(packages: list[str]) -> tuple[list[str], str]:
    python = sys.executable
    if _python_has_pip(python):
        return [python, "-m", "pip", "install", *packages], "pip"

    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", python, *packages], "uv"

    raise RuntimeError(
        "Neither 'python -m pip' nor 'uv' is available. "
        "Run: python -m ensurepip --upgrade --default-pip"
    )


def _run_install(cmd: list[str]) -> int:
    print("[cmd]", " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout)
    return completed.returncode


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.tensorrt_root).expanduser().resolve()
    cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel_dir = root / "python"
    wheels = [
        wheel_dir / f"tensorrt-10.13.3.9-{cp_tag}-none-linux_x86_64.whl",
        wheel_dir / f"tensorrt_lean-10.13.3.9-{cp_tag}-none-linux_x86_64.whl",
        wheel_dir / f"tensorrt_dispatch-10.13.3.9-{cp_tag}-none-linux_x86_64.whl",
    ]
    missing = [str(w) for w in wheels if not w.exists()]
    if missing:
        print("[install] missing TensorRT wheels for this Python:")
        for item in missing:
            print("  -", item)
        return 2

    packages = [*[str(w) for w in wheels], "onnx"]
    if args.with_onnxruntime_gpu:
        packages.append("onnxruntime-gpu")

    try:
        cmd, backend = _build_install_command(packages)
    except RuntimeError as exc:
        print(f"[install] {exc}")
        return 2

    print(f"[install] backend={backend}")
    print("[install]", " ".join(cmd))
    if args.dry_run:
        return 0
    returncode = _run_install(cmd)
    if returncode != 0:
        print(f"[install] failed with exit code {returncode}")
        return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
