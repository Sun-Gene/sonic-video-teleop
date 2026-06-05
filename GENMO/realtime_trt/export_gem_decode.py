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


class GEMDecodeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.endecoder = model.pipeline.endecoder

    def forward(self, pred_x_start: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Keep the TensorRT graph to ONNX-friendly tensor math. Full axis-angle
        # conversion uses torch.sinc/index_put in GENMO and is handled in the
        # runtime as a lightweight in-memory geometry postprocess.
        x = self.endecoder.denormalize(pred_x_start, "gvhmr")
        return (
            x[:, :, :126],
            x[:, :, 126:136],
            x[:, :, 136:142],
            x[:, :, 142:148],
            x[:, :, 148:151],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GEM motion decode to ONNX/TensorRT")
    parser.add_argument("--ckpt-path", default=str(GENMO_ROOT / "inputs/pretrained/gem_smpl.ckpt"))
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--onnx-out", default=str(DEFAULT_OUTPUT_ROOT / "onnx/gem_decode_w32.onnx"))
    parser.add_argument("--engine-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/gem_decode_w32.engine"))
    parser.add_argument("--metadata-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/gem_decode_metadata.json"))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_ROOT / "export_gem_decode_report.json"))
    parser.add_argument("--tensorrt-root", default=str(DEFAULT_TENSORRT_ROOT))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    add_genmo_to_syspath()
    report: dict[str, Any] = {"module": "gem_decode", "ok": False, "window_frames": args.window_frames}
    started = time.time()
    try:
        from demo_utils import load_model

        model = load_model(args.ckpt_path).cuda().eval()
        if model.pipeline.endecoder.obs_indices_dict is None:
            model.pipeline.endecoder.build_obs_indices_dict()
        wrapper = GEMDecodeWrapper(model).cuda().eval()
        L = args.window_frames
        pred_x_start = torch.zeros(1, L, model.endecoder.get_motion_dim(), device="cuda")

        onnx_out = Path(args.onnx_out).expanduser().resolve()
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            (pred_x_start,),
            str(onnx_out),
            input_names=["pred_x_start"],
            output_names=[
                "body_pose_r6d",
                "betas",
                "global_orient_r6d",
                "global_orient_gv_r6d",
                "local_transl_vel",
            ],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        metadata = {
            "ok": True,
            "engine_kind": "gem_decode",
            "decode_kind": "raw_gvhmr_denormalized",
            "window_frames": L,
            "input_names": ["pred_x_start"],
            "output_names": [
                "body_pose_r6d",
                "betas",
                "global_orient_r6d",
                "global_orient_gv_r6d",
                "local_transl_vel",
            ],
        }
        write_json(args.metadata_out, metadata)
        report.update(
            {
                "ok": True,
                "onnx_path": str(onnx_out),
                "metadata_path": str(Path(args.metadata_out).expanduser().resolve()),
                "elapsed_s": round(time.time() - started, 3),
            }
        )
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
        report["elapsed_s"] = round(time.time() - started, 3)
        print(report["traceback"])
    output = write_json(args.report, report)
    print(f"[export_gem_decode] wrote {output}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
