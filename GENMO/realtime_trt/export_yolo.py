#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_trt.common import DEFAULT_OUTPUT_ROOT, build_engine_with_trtexec, find_trtexec, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the YOLOv8 person detector used by GENMO preprocessing. "
            "This uses Ultralytics export and records the produced ONNX/engine path."
        )
    )
    parser.add_argument(
        "--weights",
        default="yolov8x.pt",
        help=(
            "Ultralytics weights path/name. If the file is not local, Ultralytics may try "
            "to download it when network access is available."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "engine"],
        default="engine",
        help="Use ONNX export directly, or export ONNX then build a standard TensorRT plan with trtexec.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace in GiB")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "yolo_export"))
    parser.add_argument("--onnx-out", default=str(DEFAULT_OUTPUT_ROOT / "onnx/yolo_person.onnx"))
    parser.add_argument("--engine-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/yolo.engine"))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_ROOT / "export_yolo_report.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = {"module": "yolo", "ok": False, "format": args.format}
    try:
        from ultralytics import YOLO

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(args.weights)
        # Ultralytics' direct TensorRT export can leave a metadata-prefixed file that
        # TensorRT's Python runtime cannot deserialize. Keep the runtime path boring:
        # always export ONNX first, then let the local trtexec build the plan.
        exported_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            dynamic=args.dynamic,
            half=False,
            simplify=True,
            workspace=args.workspace,
            project=str(output_dir),
            name="yolov8_person",
        )
        exported_path = Path(exported_path).expanduser().resolve()
        if exported_path.exists():
            onnx_target = Path(args.onnx_out).expanduser().resolve()
            onnx_target.parent.mkdir(parents=True, exist_ok=True)
            if exported_path != onnx_target:
                shutil.copy2(exported_path, onnx_target)
        else:
            onnx_target = exported_path

        engine_result = None
        if args.format == "engine":
            engine_result = build_engine_with_trtexec(
                onnx_path=onnx_target,
                engine_path=args.engine_out,
                trtexec=find_trtexec(),
                fp16=args.fp16,
                workspace_mb=int(args.workspace * 1024),
            )
            target = Path(args.engine_out).expanduser().resolve()
            if not engine_result.get("ok"):
                report.update(
                    {
                        "onnx_path": str(onnx_target),
                        "engine": engine_result,
                    }
                )
                output = write_json(args.report, report)
                print(f"[export_yolo] wrote {output}")
                return 2
        else:
            target = onnx_target

        report.update(
            {
                "ok": True,
                "weights": args.weights,
                "exported_path": str(exported_path),
                "onnx_path": str(onnx_target),
                "normalized_path": str(target),
                "engine": engine_result,
                "note": "Runtime must still filter class 0/person and run tracking/smoothing.",
            }
        )
    except Exception:
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])

    output = write_json(args.report, report)
    print(f"[export_yolo] wrote {output}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
