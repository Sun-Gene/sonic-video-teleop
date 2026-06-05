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


def normalize_kp2d_trt(obs_kp2d: torch.Tensor, bbx_xys: torch.Tensor) -> torch.Tensor:
    """TensorRT-friendly normalize_kp2d.

    GENMO's helper combines bool masks with `+`, which ONNX exports as BOOL Add.
    TensorRT rejects that pattern, so this exporter uses explicit logical OR and
    casts masks to float only where numeric multiplication is needed.
    """

    obs_xy = obs_kp2d[..., :2]
    center = bbx_xys[..., :2]
    scale = bbx_xys[..., [2]]
    xy_max = center + scale / 2
    xy_min = center - scale / 2
    invisible_mask = torch.logical_or(
        torch.logical_or(obs_xy[..., 0] < xy_min[..., None, 0], obs_xy[..., 0] > xy_max[..., None, 0]),
        torch.logical_or(obs_xy[..., 1] < xy_min[..., None, 1], obs_xy[..., 1] > xy_max[..., None, 1]),
    )
    scale = scale.clamp(min=1e-2)
    normalized_obs_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)
    obs_conf = obs_kp2d[..., 2] * torch.logical_not(invisible_mask).to(obs_kp2d.dtype)
    return torch.cat([normalized_obs_xy, obs_conf[..., None]], dim=-1)


class GEMConditionWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def _normalize_if_needed(self, value: torch.Tensor, key: str) -> torch.Tensor:
        if key in self.model.normalizer_stats:
            return self.model.normalize_attr(value, key)
        return value

    def _with_exists(
        self,
        key: str,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        empty: torch.Tensor,
        exists: torch.Tensor,
        uncond_exists: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.model.model_cfg.use_cond_exists_as_input or key in self.model.no_exist_keys:
            return cond, uncond, empty
        if uncond_exists is None:
            uncond_exists = exists
        empty_exists = torch.zeros_like(exists)
        cond = self.model.cond_exists_embedder[key](torch.cat([cond, exists], dim=-1))
        uncond = self.model.cond_exists_embedder[key](torch.cat([uncond, uncond_exists], dim=-1))
        empty = self.model.cond_exists_embedder[key](torch.cat([empty, empty_exists], dim=-1))
        return cond, uncond, empty

    def forward(
        self,
        kp2d: torch.Tensor,
        bbx_xys: torch.Tensor,
        K_fullimg: torch.Tensor,
        cam_angvel: torch.Tensor,
        f_imgseq: torch.Tensor,
        has_img_mask: torch.Tensor,
        has_2d_mask: torch.Tensor,
        has_cam_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from gem.utils.cam_utils import compute_bbox_info_bedlam

        B, L = kp2d.shape[:2]
        obs = normalize_kp2d_trt(kp2d, bbx_xys)
        dtype = kp2d.dtype
        has_img = has_img_mask.to(dtype)
        has_2d_input = has_2d_mask.to(dtype)
        has_cam = has_cam_mask.to(dtype)
        j2d_visible = (kp2d[..., 2] > 0.5).to(dtype) * has_2d_input[:, :, None]
        has_2d = (j2d_visible.sum(dim=-1) > 3).to(dtype)
        obs = obs * j2d_visible[:, :, :, None]
        if self.model.model_cfg.normalize_cam_angvel:
            f_cam_angvel = (cam_angvel - self.model.cam_angvel_mean) / self.model.cam_angvel_std
        else:
            f_cam_angvel = cam_angvel
        f_cliffcam = self._normalize_if_needed(compute_bbox_info_bedlam(bbx_xys, K_fullimg), "f_cliffcam")
        f_imgseq = self._normalize_if_needed(f_imgseq, "f_imgseq")
        f_cam_angvel = self._normalize_if_needed(f_cam_angvel, "f_cam_angvel")

        f_cond_dict: dict[str, torch.Tensor] = {}
        f_uncond_dict: dict[str, torch.Tensor] = {}
        f_empty_dict: dict[str, torch.Tensor] = {}
        zero_latent = torch.zeros(B, L, self.model.latent_dim, dtype=dtype, device=kp2d.device)

        for key in self.model.pipeline.args.in_attr:
            if key == "obs":
                visible = (obs[..., [2]] > 0.5).to(dtype)
                f_obs = self.model.learned_pos_linear(obs[..., :2])
                learned = self.model.learned_pos_params.view(1, 1, -1, 32).expand(B, L, -1, -1)
                f_obs = f_obs * visible + learned * (1.0 - visible)
                f_obs_empty = learned
                f_obs = self.model.embed_noisyobs(f_obs.reshape(B, L, -1))
                f_obs_empty = self.model.embed_noisyobs(f_obs_empty.reshape(B, L, -1))
                exists = (j2d_visible.sum(dim=-1, keepdim=True) > 0).to(dtype)
                empty_exists = torch.zeros_like(exists)
                cond, uncond, empty = self._with_exists(
                    key,
                    f_obs,
                    f_obs,
                    f_obs_empty,
                    exists,
                    uncond_exists=exists,
                )
                if self.model.model_cfg.use_cond_exists_as_input and key not in self.model.no_exist_keys:
                    empty = self.model.cond_exists_embedder[key](
                        torch.cat([f_obs_empty, empty_exists], dim=-1)
                    )
                f_cond_dict[key], f_uncond_dict[key], f_empty_dict[key] = cond, uncond, empty
            elif key == "f_cliffcam":
                f_value = self.model.cliffcam_embedder(f_cliffcam)
                exists = has_2d[:, :, None]
                cond, uncond, empty = self._with_exists(
                    key,
                    f_value * exists,
                    f_value * exists,
                    torch.zeros_like(f_value),
                    exists,
                )
                f_cond_dict[key], f_uncond_dict[key], f_empty_dict[key] = cond, uncond, empty
            elif key == "f_imgseq":
                f_value = self.model.imgseq_embedder(f_imgseq)
                exists = has_img[:, :, None]
                cond, uncond, empty = self._with_exists(
                    key,
                    f_value * exists,
                    f_value * exists,
                    torch.zeros_like(f_value),
                    exists,
                )
                f_cond_dict[key], f_uncond_dict[key], f_empty_dict[key] = cond, uncond, empty
            elif key == "f_cam_angvel":
                f_value = self.model.cam_angvel_embedder(f_cam_angvel)
                exists = has_cam[:, :, None]
                cond, uncond, empty = self._with_exists(
                    key,
                    f_value * exists,
                    f_value * exists,
                    torch.zeros_like(f_value),
                    exists,
                )
                f_cond_dict[key], f_uncond_dict[key], f_empty_dict[key] = cond, uncond, empty
            elif key == "encoded_music":
                exists = torch.zeros(B, L, 1, dtype=dtype, device=kp2d.device)
                cond, uncond, empty = self._with_exists(key, zero_latent, zero_latent, zero_latent, exists)
                f_cond_dict[key], f_uncond_dict[key], f_empty_dict[key] = cond, uncond, empty
            elif key == "encoded_audio":
                exists = torch.zeros(B, L, 1, dtype=dtype, device=kp2d.device)
                cond, uncond, empty = self._with_exists(key, zero_latent, zero_latent, zero_latent, exists)
                f_cond_dict[key], f_uncond_dict[key], f_empty_dict[key] = cond, uncond, empty
            else:
                # The current video-teleop runtime does not feed other modalities.
                f_cond_dict[key] = zero_latent
                f_uncond_dict[key] = zero_latent
                f_empty_dict[key] = zero_latent

        f_cond = None
        f_uncond = None
        f_empty = None
        for key in self.model.pipeline.args.in_attr:
            f_cond = f_cond_dict[key] if f_cond is None else f_cond + f_cond_dict[key]
            f_uncond = f_uncond_dict[key] if f_uncond is None else f_uncond + f_uncond_dict[key]
            f_empty = f_empty_dict[key] if f_empty is None else f_empty + f_empty_dict[key]
        assert f_cond is not None and f_uncond is not None and f_empty is not None
        return f_cond, f_uncond, f_empty


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GEM condition assembly to ONNX/TensorRT")
    parser.add_argument("--ckpt-path", default=str(GENMO_ROOT / "inputs/pretrained/gem_smpl.ckpt"))
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--onnx-out", default=str(DEFAULT_OUTPUT_ROOT / "onnx/gem_condition_w32.onnx"))
    parser.add_argument("--engine-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/gem_condition_w32.engine"))
    parser.add_argument("--metadata-out", default=str(DEFAULT_OUTPUT_ROOT / "engines/gem_condition_metadata.json"))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_ROOT / "export_gem_condition_report.json"))
    parser.add_argument("--tensorrt-root", default=str(DEFAULT_TENSORRT_ROOT))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    add_genmo_to_syspath()
    report: dict[str, Any] = {"module": "gem_condition", "ok": False, "window_frames": args.window_frames}
    started = time.time()
    try:
        from demo_utils import load_model

        model = load_model(args.ckpt_path).cuda().eval()
        if model.endecoder.obs_indices_dict is None:
            model.endecoder.build_obs_indices_dict()
        wrapper = GEMConditionWrapper(model).cuda().eval()
        L = args.window_frames
        kp2d = torch.zeros(1, L, 17, 3, device="cuda")
        kp2d[..., 2] = 1.0
        bbx_xys = torch.tensor([[[320.0, 240.0, 480.0]]], device="cuda").expand(1, L, 3).contiguous()
        K_fullimg = torch.eye(3, device="cuda").view(1, 1, 3, 3).expand(1, L, 3, 3).contiguous()
        K_fullimg[..., 0, 0] = 640.0
        K_fullimg[..., 1, 1] = 640.0
        K_fullimg[..., 0, 2] = 320.0
        K_fullimg[..., 1, 2] = 240.0
        cam_angvel = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], device="cuda").expand(1, L, 6).contiguous()
        f_imgseq = torch.zeros(1, L, 1024, device="cuda")
        has_img_mask = torch.ones(1, L, device="cuda")
        has_2d_mask = torch.ones(1, L, device="cuda")
        has_cam_mask = torch.zeros(1, L, device="cuda")

        onnx_out = Path(args.onnx_out).expanduser().resolve()
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            (kp2d, bbx_xys, K_fullimg, cam_angvel, f_imgseq, has_img_mask, has_2d_mask, has_cam_mask),
            str(onnx_out),
            input_names=[
                "kp2d",
                "bbx_xys",
                "K_fullimg",
                "cam_angvel",
                "f_imgseq",
                "has_img_mask",
                "has_2d_mask",
                "has_cam_mask",
            ],
            output_names=["f_cond", "f_uncond", "f_empty"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        metadata = {
            "ok": True,
            "engine_kind": "gem_condition",
            "window_frames": L,
            "input_names": [
                "kp2d",
                "bbx_xys",
                "K_fullimg",
                "cam_angvel",
                "f_imgseq",
                "has_img_mask",
                "has_2d_mask",
                "has_cam_mask",
            ],
            "output_names": ["f_cond", "f_uncond", "f_empty"],
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
    print(f"[export_gem_condition] wrote {output}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
