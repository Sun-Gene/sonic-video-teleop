from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any


GENMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROOT_ROOT = GENMO_ROOT.parent / "GR00T-WholeBodyControl"
DEFAULT_TENSORRT_ROOT = GENMO_ROOT.parent / "TensorRT-10.13.3.9"
DEFAULT_OUTPUT_ROOT = DEFAULT_GROOT_ROOT / "outputs" / "realtime_trt"


def add_genmo_to_syspath() -> None:
    for path in (GENMO_ROOT, GENMO_ROOT / "scripts" / "demo"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def find_trtexec(tensorrt_root: str | Path | None = None) -> str | None:
    candidates: list[Path] = []
    if tensorrt_root:
        root = Path(tensorrt_root).expanduser().resolve()
        candidates.extend(
            [
                root / "bin" / "trtexec",
                root / "samples" / "trtexec",
                root / "targets" / "x86_64-linux-gnu" / "bin" / "trtexec",
            ]
        )
    candidates.extend(
        [
            DEFAULT_TENSORRT_ROOT / "bin" / "trtexec",
            DEFAULT_TENSORRT_ROOT / "samples" / "trtexec",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("trtexec")


def run_command(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("[cmd]", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def build_engine_with_trtexec(
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    trtexec: str | None,
    fp16: bool = True,
    workspace_mb: int = 4096,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    if trtexec is None:
        return {"ok": False, "error": "trtexec not found"}

    onnx_path = Path(onnx_path).expanduser().resolve()
    engine_path = Path(engine_path).expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")
    if extra_args:
        cmd.extend(extra_args)

    try:
        completed = run_command(cmd, check=True)
        return {
            "ok": True,
            "engine_path": str(engine_path),
            "log": completed.stdout[-12000:],
        }
    except subprocess.CalledProcessError as exc:
        return {
            "ok": False,
            "engine_path": str(engine_path),
            "returncode": exc.returncode,
            "log": (exc.stdout or "")[-12000:],
            "traceback": traceback.format_exc(),
        }
    except Exception:
        return {
            "ok": False,
            "engine_path": str(engine_path),
            "traceback": traceback.format_exc(),
        }


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
        return {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
            "path": getattr(module, "__file__", "unknown"),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
