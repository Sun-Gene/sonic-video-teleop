#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_trt.common import DEFAULT_OUTPUT_ROOT, write_json
from realtime_trt.runtime import EnginePaths, MissingEngineError, RealtimeTRTPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile the TensorRT realtime pipeline readiness")
    parser.add_argument("--video", required=True)
    parser.add_argument("--engine-root", default=str(DEFAULT_OUTPUT_ROOT / "engines"))
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--observed-frames", type=int, default=8)
    parser.add_argument("--overlap-frames", type=int, default=8)
    parser.add_argument("--latency-budget-ms", type=float, default=150.0)
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_ROOT / "profile_pipeline_report.json"))
    return parser


def _read_window(video: str, count: int) -> tuple[Any, int]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    frames = []
    try:
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames, len(frames)


def main() -> int:
    args = build_parser().parse_args()
    report: dict[str, Any] = {"ok": False, "video": args.video, "engine_root": args.engine_root}
    try:
        frames, count = _read_window(args.video, args.window_frames)
        report["input_window_frames"] = count
        engine_paths = EnginePaths.from_root(args.engine_root)
        report["engines"] = engine_paths.as_dict()
        started = time.monotonic()
        try:
            _ = RealtimeTRTPipeline(
                engine_paths,
                window_frames=args.window_frames,
                observed_frames=args.observed_frames,
                overlap_frames=args.overlap_frames,
                latency_budget_ms=args.latency_budget_ms,
            )
            report["runtime_ready"] = True
            report["note"] = (
                "Engines are present. Run realtime_trt_stream.py to continue binding/runtime validation."
            )
        except MissingEngineError as exc:
            report["runtime_ready"] = False
            report["blocker"] = str(exc)
        report["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
        report["ok"] = bool(report.get("runtime_ready"))
    except Exception as exc:
        report["error"] = repr(exc)

    output = write_json(args.report, report)
    print(f"[profile] wrote {output}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
