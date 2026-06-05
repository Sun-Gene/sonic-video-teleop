#!/usr/bin/env python3
"""Persistent GENMO/GEM worker for realtime_stream.py.

The parent process runs in the SONIC environment. This worker is launched with
the GENMO Python interpreter so GENMO dependencies stay isolated. Communication
uses JSON lines:

stdin:
  {"type": "process", "job_id": 0, "video_path": "...", "output_dir": "..."}
  {"type": "shutdown"}

stdout:
  {"type": "ready", ...}
  {"type": "result", "ok": true, ...}

All GENMO logs are redirected to stderr so stdout remains machine-readable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


def _write_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _setup_imports(genmo_repo: Path) -> None:
    os.chdir(genmo_repo)
    for path in (genmo_repo, genmo_repo / "scripts" / "demo"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _save_smpl_params(pred: dict, output_dir: Path) -> Path:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "smpl_params.pt"
    save_dict: dict[str, Any] = {}
    for key_group in ("body_params_incam", "body_params_global"):
        if key_group in pred:
            save_dict[key_group] = {k: v.detach().cpu() for k, v in pred[key_group].items()}
    if "K_fullimg" in pred:
        save_dict["K_fullimg"] = pred["K_fullimg"].detach().cpu()
    torch.save(save_dict, params_path)
    return params_path


class GenmoRuntime:
    def __init__(
        self,
        *,
        genmo_repo: Path,
        ckpt_path: Path,
        hmr2_ckpt: Path,
        static_cam: bool,
        write_overlay: bool,
    ) -> None:
        _setup_imports(genmo_repo)

        import torch
        from demo_smpl_hpe import assemble_data
        from demo_utils import (
            CocoPoseExtractor,
            detect_and_track,
            get_camera_static,
            get_image_features,
            load_model,
            render_2d_keypoints,
            run_inference,
        )
        from gem.utils.video_io_utils import get_video_lwh, read_video_np

        self._torch = torch
        self._assemble_data = assemble_data
        self._CocoPoseExtractor = CocoPoseExtractor
        self._detect_and_track = detect_and_track
        self._get_camera_static = get_camera_static
        self._get_image_features = get_image_features
        self._render_2d_keypoints = render_2d_keypoints
        self._run_inference = run_inference
        self._get_video_lwh = get_video_lwh
        self._read_video_np = read_video_np

        self.hmr2_ckpt = hmr2_ckpt
        self.static_cam = static_cam
        self.write_overlay = write_overlay

        _log(f"[genmo_worker] torch={torch.__version__} cuda={torch.cuda.is_available()}")
        _log(f"[genmo_worker] loading GEM model once: {ckpt_path}")
        self.model = load_model(str(ckpt_path))
        _log("[genmo_worker] GEM model ready")

    def process(self, job: dict[str, Any]) -> dict[str, Any]:
        import torch

        job_id = int(job["job_id"])
        video_path = Path(job["video_path"])
        output_dir = Path(job["output_dir"])
        preprocess_dir = output_dir / "preprocess"
        output_dir.mkdir(parents=True, exist_ok=True)
        preprocess_dir.mkdir(parents=True, exist_ok=True)

        stage_times: dict[str, float] = {}
        started = time.monotonic()

        _log(f"[genmo_worker] job={job_id} video={video_path}")

        t_stage = time.monotonic()
        L, W, H = self._get_video_lwh(str(video_path))
        bbx_xys = self._detect_and_track(str(video_path), str(preprocess_dir))
        stage_times["detect_track_s"] = time.monotonic() - t_stage

        t_stage = time.monotonic()
        vitpose_cache = preprocess_dir / "vitpose.pt"
        if vitpose_cache.exists():
            kp2d = torch.load(vitpose_cache, map_location="cpu")
        else:
            frames_np = self._read_video_np(str(video_path))
            extractor = self._CocoPoseExtractor(device="cuda")
            kp2d = extractor.extract(frames_np, bbx_xys, batch_size=32)
            torch.save(kp2d, vitpose_cache)
            del frames_np
        stage_times["vitpose_s"] = time.monotonic() - t_stage

        t_stage = time.monotonic()
        f_imgseq, has_img_mask = self._get_image_features(
            str(video_path),
            bbx_xys,
            str(self.hmr2_ckpt),
            str(preprocess_dir),
        )
        stage_times["hmr2_s"] = time.monotonic() - t_stage

        t_stage = time.monotonic()
        R_w2c, cam_angvel, cam_tvel, K_fullimg = self._get_camera_static(L, W, H)
        data = self._assemble_data(
            kp2d=kp2d,
            bbx_xys=bbx_xys,
            K_fullimg=K_fullimg,
            cam_angvel=cam_angvel,
            cam_tvel=cam_tvel,
            R_w2c=R_w2c,
            f_imgseq=f_imgseq,
            has_img_mask=has_img_mask,
            static_cam=self.static_cam,
        )
        stage_times["assemble_s"] = time.monotonic() - t_stage

        t_stage = time.monotonic()
        pred = self._run_inference(self.model, data, static_cam=self.static_cam)
        stage_times["gem_inference_s"] = time.monotonic() - t_stage

        t_stage = time.monotonic()
        smpl_params_path = _save_smpl_params(pred, output_dir)
        if self.write_overlay:
            self._render_2d_keypoints(str(video_path), kp2d, bbx_xys, str(output_dir))
        stage_times["save_s"] = time.monotonic() - t_stage

        elapsed = time.monotonic() - started
        _log(f"[genmo_worker] job={job_id} done in {elapsed:.2f}s frames={L}")
        return {
            "type": "result",
            "ok": True,
            "job_id": job_id,
            "smpl_params_path": str(smpl_params_path),
            "output_dir": str(output_dir),
            "frames": int(L),
            "width": int(W),
            "height": int(H),
            "elapsed_s": elapsed,
            "stage_times": stage_times,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent GENMO/GEM JSONL worker")
    parser.add_argument("--genmo-repo", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--hmr2-ckpt", required=True)
    parser.add_argument("--static-cam", action="store_true")
    parser.add_argument("--write-overlay", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = GenmoRuntime(
                genmo_repo=Path(args.genmo_repo).expanduser().resolve(),
                ckpt_path=Path(args.ckpt_path).expanduser().resolve(),
                hmr2_ckpt=Path(args.hmr2_ckpt).expanduser().resolve(),
                static_cam=args.static_cam,
                write_overlay=args.write_overlay,
            )
        _write_json({"type": "ready", "ok": True})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
                if job.get("type") == "shutdown":
                    _write_json({"type": "shutdown", "ok": True})
                    return 0
                if job.get("type") != "process":
                    raise ValueError(f"Unsupported job type: {job.get('type')!r}")
                with contextlib.redirect_stdout(sys.stderr):
                    result = runtime.process(job)
                _write_json(result)
            except Exception as exc:  # keep worker alive for the next job
                _log(traceback.format_exc())
                _write_json(
                    {
                        "type": "result",
                        "ok": False,
                        "job_id": job.get("job_id") if isinstance(job, dict) else None,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        return 0
    except Exception as exc:
        _log(traceback.format_exc())
        _write_json({"type": "fatal", "ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
