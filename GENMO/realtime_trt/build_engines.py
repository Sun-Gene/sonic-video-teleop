#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_trt.common import DEFAULT_OUTPUT_ROOT, GENMO_ROOT, write_json


SCRIPT_DIR = Path(__file__).resolve().parent


def _run_step(name: str, cmd: list[str]) -> dict[str, Any]:
    started = time.time()
    print(f"[build] {name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(GENMO_ROOT),
        bufsize=1,
    )
    log_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_parts.append(line)
        if len(log_parts) > 800:
            del log_parts[: len(log_parts) - 800]
    returncode = proc.wait()
    ok = returncode == 0
    log_tail = "".join(log_parts)[-12000:]
    if not ok:
        print(f"[build] {name} failed with exit code {returncode}", flush=True)
    return {
        "name": name,
        "ok": ok,
        "returncode": returncode,
        "elapsed_s": round(time.time() - started, 3),
        "log_tail": log_tail,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrate GENMO TensorRT export/build steps")
    parser.add_argument("--example-video", default=None, help="Required for GEM denoiser export")
    parser.add_argument("--ckpt-path", default=str(GENMO_ROOT / "inputs/pretrained/gem_smpl.ckpt"))
    parser.add_argument(
        "--hmr2-ckpt",
        default=str(GENMO_ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-vitpose", action="store_true")
    parser.add_argument("--skip-hmr2", action="store_true")
    parser.add_argument("--skip-gem", action="store_true")
    parser.add_argument("--skip-gem-condition", action="store_true")
    parser.add_argument("--skip-gem-decode", action="store_true")
    parser.add_argument(
        "--legacy-captured-gem",
        action="store_true",
        help="Build the old captured-condition GEM engine instead of the explicit-condition runtime engine.",
    )
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--build-engines", action="store_true")
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_ROOT / "build_engines_summary.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    common_flags = ["--fp16"] if args.fp16 else []
    build_flag = ["--build-engine"] if args.build_engines else []
    steps: list[tuple[str, list[str]]] = [
        (
            "diagnose",
            [
                py,
                str(SCRIPT_DIR / "diagnose_env.py"),
                "--output",
                str(output_root / "diagnostics.json"),
                "--ckpt-path",
                args.ckpt_path,
                "--hmr2-ckpt",
                args.hmr2_ckpt,
            ],
        )
    ]
    if not args.skip_yolo:
        steps.append(
            (
                "yolo",
                [
                    py,
                    str(SCRIPT_DIR / "export_yolo.py"),
                    "--format",
                    "engine" if args.build_engines else "onnx",
                    "--report",
                    str(output_root / "export_yolo_report.json"),
                    *common_flags,
                ],
            )
        )
    if not args.skip_vitpose:
        steps.append(
            (
                "vitpose",
                [
                    py,
                    str(SCRIPT_DIR / "export_vitpose.py"),
                    "--report",
                    str(output_root / "export_vitpose_report.json"),
                    *build_flag,
                    *common_flags,
                ],
            )
        )
    if not args.skip_hmr2:
        steps.append(
            (
                "hmr2",
                [
                    py,
                    str(SCRIPT_DIR / "export_hmr2.py"),
                    "--report",
                    str(output_root / "export_hmr2_report.json"),
                    "--ckpt",
                    args.hmr2_ckpt,
                    *build_flag,
                    *common_flags,
                ],
            )
        )
    if not args.skip_gem:
        if not args.example_video:
            print("[build] skipping GEM denoiser export because --example-video was not provided")
        else:
            steps.append(
                (
                    "gem_denoiser_legacy" if args.legacy_captured_gem else "gem_denoiser_explicit",
                    [
                        py,
                        str(
                            SCRIPT_DIR
                            / ("export_gem_denoiser.py" if args.legacy_captured_gem else "export_gem_denoiser_explicit.py")
                        ),
                        "--example-video",
                        args.example_video,
                        "--ckpt-path",
                        args.ckpt_path,
                        "--hmr2-ckpt",
                        args.hmr2_ckpt,
                        "--report",
                        str(
                            output_root
                            / (
                                "export_gem_denoiser_report.json"
                                if args.legacy_captured_gem
                                else "export_gem_denoiser_explicit_report.json"
                            )
                        ),
                        *([] if args.legacy_captured_gem else ["--window-frames", str(args.window_frames)]),
                        *build_flag,
                        *common_flags,
                    ],
                )
            )
    if not args.skip_gem_condition:
        steps.append(
            (
                "gem_condition",
                [
                    py,
                    str(SCRIPT_DIR / "export_gem_condition.py"),
                    "--ckpt-path",
                    args.ckpt_path,
                    "--report",
                    str(output_root / "export_gem_condition_report.json"),
                    "--window-frames",
                    str(args.window_frames),
                    *build_flag,
                    *common_flags,
                ],
            )
        )
    if not args.skip_gem_decode:
        steps.append(
            (
                "gem_decode",
                [
                    py,
                    str(SCRIPT_DIR / "export_gem_decode.py"),
                    "--ckpt-path",
                    args.ckpt_path,
                    "--report",
                    str(output_root / "export_gem_decode_report.json"),
                    "--window-frames",
                    str(args.window_frames),
                    *build_flag,
                    *common_flags,
                ],
            )
        )

    results = [_run_step(name, cmd) for name, cmd in steps]
    summary = {"ok": all(item["ok"] for item in results), "steps": results}
    output = write_json(args.report, summary)
    print(f"[build] wrote {output}")
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
