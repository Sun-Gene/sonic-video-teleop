#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_trt.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TENSORRT_ROOT,
    GENMO_ROOT,
    add_genmo_to_syspath,
    build_engine_with_trtexec,
    find_trtexec,
    write_json,
)


class HMR2TensorWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model({"img": img})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export HMR2 feature extractor to ONNX/TensorRT")
    parser.add_argument(
        "--ckpt",
        default=str(GENMO_ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--onnx-out", default=str(DEFAULT_OUTPUT_ROOT / "onnx/hmr2_b8.onnx"))
    parser.add_argument("--engine-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/hmr2_b8.engine"))
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--tensorrt-root", default=str(DEFAULT_TENSORRT_ROOT))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_ROOT / "export_hmr2_report.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    add_genmo_to_syspath()
    report = {"module": "hmr2", "ok": False}
    try:
        from gem.network.hmr2 import load_hmr2

        model = HMR2TensorWrapper(load_hmr2(args.ckpt)).cuda().eval()
        dummy = torch.randn(args.batch_size, 3, 256, 256, device="cuda")
        onnx_out = Path(args.onnx_out).expanduser().resolve()
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_out),
            input_names=["img"],
            output_names=["f_imgseq"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes={"img": {0: "batch"}, "f_imgseq": {0: "batch"}},
        )
        report.update({"ok": True, "onnx_path": str(onnx_out)})
        if args.build_engine:
            engine_report = build_engine_with_trtexec(
                onnx_path=onnx_out,
                engine_path=args.engine_out,
                trtexec=find_trtexec(args.tensorrt_root),
                fp16=args.fp16,
            )
            report["engine"] = engine_report
            report["ok"] = bool(engine_report.get("ok"))
    except Exception:
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
    output = write_json(args.report, report)
    print(f"[export_hmr2] wrote {output}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
