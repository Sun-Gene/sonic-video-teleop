from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import subprocess
import time
from typing import Any

import numpy as np


class MissingEngineError(RuntimeError):
    pass


@dataclass(slots=True)
class EnginePaths:
    root: Path
    yolo: Path | None = None
    vitpose: Path | None = None
    hmr2: Path | None = None
    gem_denoiser: Path | None = None
    gem_condition: Path | None = None
    gem_decode: Path | None = None

    @classmethod
    def from_root(cls, root: str | Path) -> "EnginePaths":
        root_path = Path(root).expanduser().resolve()
        return cls(
            root=root_path,
            yolo=_first_existing(root_path, ["yolo.engine", "yolov8_person.engine"]),
            vitpose=_first_existing(root_path, ["vitpose_b8.engine", "vitpose.engine"]),
            hmr2=_first_existing(root_path, ["hmr2_b8.engine", "hmr2.engine"]),
            gem_denoiser=_first_existing(
                root_path,
                [
                    "gem_denoiser_explicit_w32.engine",
                    "gem_denoiser_explicit.engine",
                    "gem_denoiser_step.engine",
                    "gem_denoiser.engine",
                ],
            ),
            gem_condition=_first_existing(root_path, ["gem_condition.engine"]),
            gem_decode=_first_existing(root_path, ["gem_decode_w32.engine", "gem_decode.engine"]),
        )

    def missing(self, *, paper_trt: bool = True) -> list[str]:
        required = {
            "yolo": self.yolo,
            "vitpose": self.vitpose,
            "hmr2": self.hmr2,
            "gem_denoiser": self.gem_denoiser,
        }
        if paper_trt:
            required["gem_condition"] = self.gem_condition
            required["gem_decode"] = self.gem_decode
        return [name for name, path in required.items() if path is None or not path.exists()]

    def as_dict(self) -> dict[str, str | None]:
        return {
            "root": str(self.root),
            "yolo": str(self.yolo) if self.yolo else None,
            "vitpose": str(self.vitpose) if self.vitpose else None,
            "hmr2": str(self.hmr2) if self.hmr2 else None,
            "gem_denoiser": str(self.gem_denoiser) if self.gem_denoiser else None,
            "gem_condition": str(self.gem_condition) if self.gem_condition else None,
            "gem_decode": str(self.gem_decode) if self.gem_decode else None,
        }


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    matches = []
    for suffix in names:
        stem = Path(suffix).stem
        matches.extend(root.glob(f"*{stem}*.engine"))
    return sorted(matches)[0] if matches else None


@dataclass(slots=True)
class LatencyStats:
    values_ms: list[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        self.values_ms.append(float(value_ms))
        if len(self.values_ms) > 2000:
            del self.values_ms[: len(self.values_ms) - 2000]

    def snapshot(self) -> dict[str, float]:
        if not self.values_ms:
            return {"count": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        arr = np.asarray(self.values_ms, dtype=np.float32)
        return {
            "count": float(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }


class SlidingWindowInpainter:
    """Overwrite the current prediction's overlap prefix with the previous clean motion.

    The SONIC paper describes doing this inside the diffusion denoising loop. This
    helper implements the runtime state and array operation; the TensorRT denoiser
    scheduler must call it at every denoising step for a true paper-style path.
    """

    def __init__(self, overlap_frames: int) -> None:
        self.overlap_frames = max(0, int(overlap_frames))
        self.previous_clean: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        self.previous_clean = None

    def apply(self, current: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.previous_clean is None or self.overlap_frames <= 0:
            self.previous_clean = {k: np.array(v, copy=True) for k, v in current.items()}
            return current
        for key, value in current.items():
            prev = self.previous_clean.get(key)
            if prev is None or value.ndim == 0:
                continue
            count = min(self.overlap_frames, value.shape[0], prev.shape[0])
            if count > 0:
                value[:count] = prev[-count:]
        self.previous_clean = {k: np.array(v, copy=True) for k, v in current.items()}
        return current


class PaperRealtimeTRTPipeline:
    """In-memory TensorRT runtime for the SONIC video teleop path.

    This class intentionally has no GENMO worker, no mp4 writer, and no
    smpl_params.pt bridge. It either runs with the required TensorRT engines or
    raises a clear blocker that names the missing artifact.
    """

    def __init__(
        self,
        engine_paths: EnginePaths,
        *,
        window_frames: int,
        observed_frames: int,
        overlap_frames: int,
        latency_budget_ms: float,
        target_fps: int,
        num_frames_to_send: int,
        source_fps: int,
        ddim_steps: int | None = None,
    ) -> None:
        from realtime_trt.trt_runner import TensorRTEngineRunner

        self.engine_paths = engine_paths
        self.window_frames = int(window_frames)
        self.observed_frames = int(observed_frames)
        self.overlap_frames = int(overlap_frames)
        self.latency_budget_ms = float(latency_budget_ms)
        self.target_fps = int(target_fps)
        self.num_frames_to_send = int(num_frames_to_send)
        self.source_fps = int(source_fps)
        self.ddim_steps = int(ddim_steps) if ddim_steps else 0
        self._job_id = 0
        self._previous_clean: np.ndarray | None = None
        self._metadata = _load_runtime_metadata(engine_paths.root)
        if "diffusion" not in self._metadata:
            raise MissingEngineError(
                "gem_runtime_metadata.json lacks diffusion scheduler arrays. Rebuild gem_denoiser_explicit_w32.engine "
                "with the updated exporter: cd /path/to/workspace/GENMO && source .venv/bin/activate && "
                "python realtime_trt/build_engines.py --example-video /path/to/input.mp4 --skip-yolo --skip-vitpose "
                "--skip-hmr2 --build-engines --fp16 --window-frames 32"
            )
        assert engine_paths.yolo and engine_paths.vitpose and engine_paths.hmr2
        assert engine_paths.gem_condition and engine_paths.gem_denoiser and engine_paths.gem_decode
        self.yolo = TensorRTEngineRunner(engine_paths.yolo)
        self.vitpose = TensorRTEngineRunner(engine_paths.vitpose)
        self.hmr2 = TensorRTEngineRunner(engine_paths.hmr2)
        self.condition = TensorRTEngineRunner(engine_paths.gem_condition)
        self.denoiser = TensorRTEngineRunner(engine_paths.gem_denoiser)
        self.decode = TensorRTEngineRunner(engine_paths.gem_decode)

    def readiness_report(self) -> dict[str, Any]:
        return {
            "runtime_bound": True,
            "engines_present": True,
            "mode": "paper_trt",
            "gem_denoiser_explicit": True,
            "missing_runtime_components": [],
            "reason": "All required TensorRT engines are present and the runtime path is in-memory.",
            "engines": self.engine_paths.as_dict(),
        }

    def close(self) -> None:
        for runner in (
            getattr(self, "yolo", None),
            getattr(self, "vitpose", None),
            getattr(self, "hmr2", None),
            getattr(self, "condition", None),
            getattr(self, "denoiser", None),
            getattr(self, "decode", None),
        ):
            if runner is not None:
                runner.close()

    def infer_window(self, frames_rgb: np.ndarray, capture_ts: float) -> dict[str, Any]:
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError(f"Expected frames_rgb [T,H,W,3], got {frames_rgb.shape}")
        started = time.monotonic()
        frames_rgb = _fit_window(frames_rgb, self.window_frames)
        height, width = frames_rgb.shape[1:3]
        observed_frames, observed_positions = _select_observed_frames(
            frames_rgb,
            max_frames=self.observed_frames,
        )

        t_stage = time.monotonic()
        observed_bbx_xys = self._infer_bboxes(observed_frames)
        bbx_xys = _expand_observed_sequence(
            observed_bbx_xys,
            observed_positions,
            self.window_frames,
        )
        yolo_s = time.monotonic() - t_stage

        t_stage = time.monotonic()
        observed_kp2d = self._infer_keypoints(observed_frames, observed_bbx_xys)
        kp2d = _expand_observed_sequence(
            observed_kp2d,
            observed_positions,
            self.window_frames,
        )
        vitpose_s = time.monotonic() - t_stage

        t_stage = time.monotonic()
        observed_f_imgseq = self._infer_hmr2(observed_frames, observed_bbx_xys)
        f_imgseq = _expand_observed_sequence(
            observed_f_imgseq,
            observed_positions,
            self.window_frames,
        )
        hmr2_s = time.monotonic() - t_stage

        t_stage = time.monotonic()
        K_fullimg, cam_angvel = _static_camera_tensors(self.window_frames, width, height)
        masks = np.ones((1, self.window_frames), dtype=np.float32)
        condition_out = self.condition.infer(
            {
                "kp2d": kp2d[None].astype(np.float32),
                "bbx_xys": bbx_xys[None].astype(np.float32),
                "K_fullimg": K_fullimg[None].astype(np.float32),
                "cam_angvel": cam_angvel[None].astype(np.float32),
                "f_imgseq": f_imgseq[None].astype(np.float32),
                "has_img_mask": masks,
                "has_2d_mask": masks,
                "has_cam_mask": np.zeros_like(masks),
            }
        )
        condition_s = time.monotonic() - t_stage

        t_stage = time.monotonic()
        pred_x_start = self._ddim_loop(condition_out)
        ddim_s = time.monotonic() - t_stage

        t_stage = time.monotonic()
        decode_out = self.decode.infer(
            {
                "pred_x_start": pred_x_start.astype(np.float32),
            }
        )
        pose_data = _pose_data_from_decode_outputs(
            decode_out=decode_out,
            cam_angvel=cam_angvel,
            source_fps=self.source_fps,
            target_fps=self.target_fps,
            num_frames_to_send=self.num_frames_to_send,
        )
        decode_s = time.monotonic() - t_stage

        total_s = time.monotonic() - started
        return {
            "pose_data": pose_data,
            "mode": "paper_trt",
            "latency_s": time.monotonic() - capture_ts,
            "stage_times": {
                "yolo_s": yolo_s,
                "vitpose_s": vitpose_s,
                "hmr2_s": hmr2_s,
                "condition_s": condition_s,
                "ddim_s": ddim_s,
                "decode_s": decode_s,
                "total_s": total_s,
                "observed_frames": float(len(observed_frames)),
            },
        }

    def _infer_bboxes(self, frames_rgb: np.ndarray) -> np.ndarray:
        boxes = []
        for frame in frames_rgb:
            inp, scale, pad = _letterbox_chw(frame, self.yolo.input_names[0], self.yolo)
            output = self.yolo.infer({self.yolo.input_names[0]: inp})
            boxes.append(_decode_yolo_person_box(output, frame.shape[1], frame.shape[0], scale, pad))
        return _smooth_bboxes(np.asarray(boxes, dtype=np.float32))

    def _infer_keypoints(self, frames_rgb: np.ndarray, bbx_xys: np.ndarray) -> np.ndarray:
        crops = _vitpose_crops(frames_rgb, bbx_xys)
        heatmaps = _batched_trt_infer(self.vitpose, self.vitpose.input_names[0], crops, self.vitpose.output_names[0])
        return _heatmaps_to_keypoints(heatmaps, bbx_xys).astype(np.float32)

    def _infer_hmr2(self, frames_rgb: np.ndarray, bbx_xys: np.ndarray) -> np.ndarray:
        crops = _hmr2_crops(frames_rgb, bbx_xys)
        features = _batched_trt_infer(self.hmr2, self.hmr2.input_names[0], crops, self.hmr2.output_names[0])
        return features[: len(frames_rgb)].astype(np.float32)

    def _ddim_loop(self, condition_out: dict[str, np.ndarray]) -> np.ndarray:
        diffusion = self._metadata["diffusion"]
        alphas = np.asarray(diffusion["alphas_cumprod"], dtype=np.float32)
        alphas_prev = np.asarray(diffusion["alphas_cumprod_prev"], dtype=np.float32)
        sqrt_recip = np.asarray(diffusion["sqrt_recip_alphas_cumprod"], dtype=np.float32)
        sqrt_recipm1 = np.asarray(diffusion["sqrt_recipm1_alphas_cumprod"], dtype=np.float32)
        xt = np.zeros(tuple(self._metadata["shapes"]["xt"]), dtype=np.float32)
        f_cond = condition_out["f_cond"].astype(np.float32)
        f_uncond = condition_out["f_uncond"].astype(np.float32)
        f_empty = condition_out["f_empty"].astype(np.float32)
        pred_x_start = xt
        num_timesteps = int(diffusion["num_timesteps"])
        if 0 < self.ddim_steps < num_timesteps:
            timestep_indices = np.linspace(
                num_timesteps - 1,
                0,
                self.ddim_steps,
                dtype=np.int64,
            )
        else:
            timestep_indices = np.arange(num_timesteps - 1, -1, -1, dtype=np.int64)
        for step_pos, index in enumerate(timestep_indices):
            t = np.asarray([index], dtype=np.int64)
            denoised = self.denoiser.infer(
                {
                    "xt": xt,
                    "timesteps": t,
                    "f_cond": f_cond,
                    "f_uncond": f_uncond,
                    "f_empty": f_empty,
                }
            )
            pred_x_start = denoised["pred_x_start"].astype(np.float32)
            if self._previous_clean is not None and self.overlap_frames > 0:
                count = min(self.overlap_frames, pred_x_start.shape[1], self._previous_clean.shape[1])
                pred_x_start[:, :count] = self._previous_clean[:, -count:]
            eps = (sqrt_recip[index] * xt - pred_x_start) / max(float(sqrt_recipm1[index]), 1e-6)
            if step_pos + 1 < len(timestep_indices):
                alpha_prev = float(alphas[int(timestep_indices[step_pos + 1])])
            else:
                alpha_prev = float(alphas_prev[0])
            mean = pred_x_start * np.sqrt(alpha_prev) + np.sqrt(max(1.0 - alpha_prev, 0.0)) * eps
            xt = mean.astype(np.float32)
        self._previous_clean = pred_x_start.copy()
        return pred_x_start


class RealtimeTRTPipeline:
    """TensorRT-bound runtime facade for video-to-Protocol-v3 streaming.

    The public GENMO release does not expose the complete paper runtime graph.
    This class therefore does two things:

    * validates/describes the TensorRT engines that are available now, including
      the new explicit-condition GEM denoiser engine when present;
    * provides a functional hybrid path that keeps GENMO model loading in a
      persistent GENMO worker and returns real Protocol v3 payloads instead of
      exiting with a blocker.

    The hybrid path is intentionally labeled in readiness/logs so it is not
    mistaken for the final all-TRT paper runtime.
    """

    def __init__(
        self,
        engine_paths: EnginePaths,
        *,
        window_frames: int,
        observed_frames: int,
        overlap_frames: int,
        latency_budget_ms: float,
        genmo_repo: str | Path | None = None,
        genmo_python: str | Path | None = None,
        ckpt_path: str | Path | None = None,
        hmr2_ckpt: str | Path | None = None,
        output_root: str | Path | None = None,
        target_fps: int = 50,
        num_frames_to_send: int = 5,
        source_fps: int = 30,
        fallback_hybrid: bool = False,
        ddim_steps: int | None = None,
    ) -> None:
        self.engine_paths = engine_paths
        self.fallback_hybrid = bool(fallback_hybrid)
        self.window_frames = int(window_frames)
        self.observed_frames = int(observed_frames)
        self.overlap_frames = int(overlap_frames)
        self.latency_budget_ms = float(latency_budget_ms)
        self.inpainter = SlidingWindowInpainter(overlap_frames=overlap_frames)
        repo_root = Path(__file__).resolve().parents[1]
        self.genmo_repo = Path(genmo_repo).expanduser().resolve() if genmo_repo else repo_root
        self.genmo_python = (
            _absolute_no_resolve(genmo_python)
            if genmo_python
            else self.genmo_repo / ".venv" / "bin" / "python"
        )
        self.ckpt_path = (
            Path(ckpt_path).expanduser().resolve()
            if ckpt_path
            else self.genmo_repo / "inputs" / "pretrained" / "gem_smpl.ckpt"
        )
        self.hmr2_ckpt = (
            Path(hmr2_ckpt).expanduser().resolve()
            if hmr2_ckpt
            else self.genmo_repo / "inputs" / "checkpoints" / "hmr2" / "epoch=10-step=25000-001.ckpt"
        )
        self.groot_root = self.genmo_repo.parent / "GR00T-WholeBodyControl"
        self.worker_path = self.groot_root / "local_video_realtime_bridge" / "genmo_worker.py"
        self.output_root = (
            Path(output_root).expanduser().resolve()
            if output_root
            else self.groot_root / "outputs" / "realtime_trt_runtime_windows"
        )
        self.target_fps = int(target_fps)
        self.num_frames_to_send = int(num_frames_to_send)
        self.source_fps = int(source_fps)
        self.ddim_steps = int(ddim_steps) if ddim_steps else 0
        self._job_id = 0
        self._worker: subprocess.Popen[str] | None = None
        self._engine_descriptions: dict[str, Any] | None = None
        self._paper: PaperRealtimeTRTPipeline | None = None
        missing = engine_paths.missing(paper_trt=not self.fallback_hybrid)
        if missing:
            raise MissingEngineError(
                "Missing TensorRT engines: "
                + ", ".join(missing)
                + f". Looked under {engine_paths.root}. Run GENMO/realtime_trt/build_engines.py first. "
                + self._required_build_command()
            )
        if not self.fallback_hybrid:
            self._paper = PaperRealtimeTRTPipeline(
                engine_paths,
                window_frames=self.window_frames,
                observed_frames=self.observed_frames,
                overlap_frames=self.overlap_frames,
                latency_budget_ms=self.latency_budget_ms,
                target_fps=self.target_fps,
                num_frames_to_send=self.num_frames_to_send,
                source_fps=self.source_fps,
                ddim_steps=self.ddim_steps,
            )

    def readiness_report(self) -> dict[str, Any]:
        explicit = self.engine_paths.gem_denoiser is not None and "explicit" in self.engine_paths.gem_denoiser.name
        if not self.fallback_hybrid:
            assert self._paper is not None
            return self._paper.readiness_report()
        mode = "hybrid_trt_bound"
        return {
            "runtime_bound": True,
            "engines_present": True,
            "mode": mode,
            "gem_denoiser_explicit": explicit,
            "reason": (
                "Runtime can now produce Protocol v3 payloads through a persistent GENMO worker. "
                "TensorRT engines are validated and the explicit GEM denoiser exporter/runner are present. "
                "The current executable path is hybrid until GENMO condition assembly, DDIM scheduling, "
                "and SMPL decode are fully moved to TensorRT."
            ),
            "missing_runtime_components": [] if explicit else [
                "Build gem_denoiser_explicit_w32.engine for runtime condition inputs",
            ],
            "engines": self.engine_paths.as_dict(),
            "worker_python": str(self.genmo_python),
            "worker_path": str(self.worker_path),
        }

    def infer_window(self, frames_rgb: np.ndarray, capture_ts: float) -> dict[str, Any]:
        if not self.fallback_hybrid:
            assert self._paper is not None
            self._paper.source_fps = self.source_fps
            return self._paper.infer_window(frames_rgb, capture_ts)
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError(f"Expected frames_rgb [T,H,W,3], got {frames_rgb.shape}")

        started = time.monotonic()
        self.output_root.mkdir(parents=True, exist_ok=True)
        job_id = self._job_id
        self._job_id += 1
        video_path = self.output_root / "window_videos" / f"window_{job_id:06d}.mp4"
        output_dir = self.output_root / "genmo_windows" / f"window_{job_id:06d}"

        t_stage = time.monotonic()
        _write_rgb_video(frames_rgb, video_path, fps=self.source_fps)
        write_video_s = time.monotonic() - t_stage

        t_stage = time.monotonic()
        worker_result = self._run_worker_job(job_id, video_path, output_dir)
        worker_s = time.monotonic() - t_stage

        t_stage = time.monotonic()
        sequence = _load_bridge_sequence(
            smpl_params_path=worker_result["smpl_params_path"],
            source_fps=self.source_fps,
            target_fps=self.target_fps,
        )
        pose_data = _build_pose_window(
            sequence=sequence,
            window_start=0,
            num_frames_to_send=self.num_frames_to_send,
            frame_index_start=0,
        )
        bridge_s = time.monotonic() - t_stage

        latency_s = time.monotonic() - capture_ts
        total_s = time.monotonic() - started
        return {
            "pose_data": pose_data,
            "mode": "hybrid_trt_bound",
            "smpl_params_path": worker_result["smpl_params_path"],
            "latency_s": latency_s,
            "stage_times": {
                "write_video_s": write_video_s,
                "worker_roundtrip_s": worker_s,
                "bridge_s": bridge_s,
                "total_s": total_s,
                **{
                    f"genmo_{k}": float(v)
                    for k, v in worker_result.get("stage_times", {}).items()
                },
            },
        }

    def close(self) -> None:
        if self._paper is not None:
            self._paper.close()
        if self._worker is None:
            return
        try:
            if self._worker.stdin and self._worker.poll() is None:
                self._worker.stdin.write('{"type":"shutdown"}\n')
                self._worker.stdin.flush()
            self._worker.wait(timeout=5.0)
        except Exception:
            if self._worker.poll() is None:
                self._worker.kill()
        finally:
            self._worker = None

    def describe_engines(self) -> dict[str, Any]:
        if self._engine_descriptions is not None:
            return self._engine_descriptions
        descriptions: dict[str, Any] = {}
        try:
            from realtime_trt.trt_runner import TensorRTEngineRunner

            for name, path in self.engine_paths.as_dict().items():
                if name == "root" or not path:
                    continue
                runner = TensorRTEngineRunner(path)
                descriptions[name] = runner.describe()
                runner.close()
        except Exception as exc:
            descriptions["error"] = repr(exc)
        self._engine_descriptions = descriptions
        return descriptions

    def _required_build_command(self) -> str:
        return (
            "Required command: cd /path/to/workspace/GENMO && source .venv/bin/activate && "
            "python realtime_trt/build_engines.py --example-video /path/to/input.mp4 "
            "--build-engines --fp16 --window-frames 32"
        )

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        if not self.worker_path.exists():
            raise RuntimeError(f"GENMO worker not found: {self.worker_path}")
        cmd = [
            str(self.genmo_python),
            str(self.worker_path),
            "--genmo-repo",
            str(self.genmo_repo),
            "--ckpt-path",
            str(self.ckpt_path),
            "--hmr2-ckpt",
            str(self.hmr2_ckpt),
        ]
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        self._worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            cwd=str(self.groot_root),
            env=env,
        )
        ready = self._read_worker_json()
        if not ready.get("ok") or ready.get("type") != "ready":
            raise RuntimeError(
                "GENMO worker failed to start: "
                f"{ready}\n"
                "Check the GENMO interpreter directly with:\n"
                f"  {self.genmo_python} -c \"import sys, torch; print(sys.prefix); print(torch.__version__)\""
            )
        print("[realtime_trt] GENMO hybrid worker ready", flush=True)
        return self._worker

    def _run_worker_job(self, job_id: int, video_path: Path, output_dir: Path) -> dict[str, Any]:
        proc = self._ensure_worker()
        assert proc.stdin is not None
        request = {
            "type": "process",
            "job_id": job_id,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
        }
        proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        proc.stdin.flush()
        response = self._read_worker_json()
        if not response.get("ok"):
            raise RuntimeError(f"GENMO worker failed: {response.get('error')}\n{response.get('traceback', '')}")
        return response

    def _read_worker_json(self) -> dict[str, Any]:
        proc = self._worker
        if proc is None or proc.stdout is None:
            raise RuntimeError("GENMO worker is not running")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("GENMO worker exited before returning JSON")
        return json.loads(line)


def _write_rgb_video(frames_rgb: np.ndarray, output: Path, fps: int) -> None:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _absolute_no_resolve(path: str | Path) -> Path:
    """Return an absolute path while preserving virtualenv symlinks.

    uv-created virtualenvs often make .venv/bin/python a symlink to a shared
    interpreter. Resolving that symlink bypasses the virtualenv site-packages,
    so subprocesses lose packages such as torch. os.path.abspath preserves the
    symlink path while still making it safe to execute from another cwd.
    """

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _load_bridge_sequence(*, smpl_params_path: str | Path, source_fps: int, target_fps: int):
    from gear_sonic.utils.teleop.genmo_bridge import load_genmo_bridge_sequence

    return load_genmo_bridge_sequence(
        smpl_params_path=smpl_params_path,
        source_fps=source_fps,
        target_fps=target_fps,
    )


def _build_pose_window(*, sequence: Any, window_start: int, num_frames_to_send: int, frame_index_start: int):
    from gear_sonic.utils.teleop.genmo_bridge import build_pose_window

    return build_pose_window(
        sequence=sequence,
        window_start=window_start,
        num_frames_to_send=num_frames_to_send,
        frame_index_start=frame_index_start,
    )


def _load_runtime_metadata(engine_root: Path) -> dict[str, Any]:
    path = engine_root / "gem_runtime_metadata.json"
    if not path.exists():
        raise MissingEngineError(f"Missing GEM runtime metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_window(frames_rgb: np.ndarray, window_frames: int) -> np.ndarray:
    if frames_rgb.shape[0] == window_frames:
        return frames_rgb
    if frames_rgb.shape[0] > window_frames:
        return frames_rgb[-window_frames:]
    pad = np.repeat(frames_rgb[-1:], window_frames - frames_rgb.shape[0], axis=0)
    return np.concatenate([frames_rgb, pad], axis=0)


def _select_observed_frames(frames_rgb: np.ndarray, *, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    length = int(frames_rgb.shape[0])
    count = min(max(1, int(max_frames)), length)
    positions = np.linspace(0, length - 1, count, dtype=np.float32)
    indices = np.clip(np.rint(positions).astype(np.int64), 0, length - 1)
    return frames_rgb[indices], positions


def _expand_observed_sequence(values: np.ndarray, positions: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    if values.shape[0] == target_length:
        return values.astype(np.float32, copy=False)
    if values.shape[0] == 1:
        return np.repeat(values, target_length, axis=0).astype(np.float32)
    target_positions = np.arange(target_length, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    expanded = np.empty((target_length, flat.shape[1]), dtype=np.float32)
    for dim in range(flat.shape[1]):
        expanded[:, dim] = np.interp(target_positions, positions, flat[:, dim]).astype(np.float32)
    return expanded.reshape((target_length,) + values.shape[1:])


def _binding_shape(runner: Any, name: str) -> tuple[int, ...]:
    for binding in runner.bindings:
        if binding.name == name:
            return tuple(binding.shape)
    raise KeyError(name)


def _letterbox_chw(frame_rgb: np.ndarray, input_name: str, runner: Any) -> tuple[np.ndarray, float, tuple[float, float]]:
    import cv2

    shape = _binding_shape(runner, input_name)
    height, width = int(shape[-2]), int(shape[-1])
    src_h, src_w = frame_rgb.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    tensor = canvas.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None]
    return tensor, scale, (float(pad_x), float(pad_y))


def _decode_yolo_person_box(
    outputs: dict[str, np.ndarray],
    width: int,
    height: int,
    scale: float,
    pad: tuple[float, float],
) -> np.ndarray:
    out = next(iter(outputs.values()))
    pred = np.asarray(out, dtype=np.float32)
    pred = np.squeeze(pred)
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.ndim != 2 or pred.shape[1] < 5:
        return _default_bbox(width, height)
    scores = pred[:, 4] if pred.shape[1] == 6 else pred[:, 4]
    idx = int(np.argmax(scores))
    if float(scores[idx]) < 0.05:
        return _default_bbox(width, height)
    cx, cy, bw, bh = pred[idx, :4]
    pad_x, pad_y = pad
    cx = (cx - pad_x) / scale
    cy = (cy - pad_y) / scale
    bw = bw / scale
    bh = bh / scale
    size = max(float(bw), float(bh)) * 1.15
    cx = float(np.clip(cx, 0, width - 1))
    cy = float(np.clip(cy, 0, height - 1))
    size = float(np.clip(size, 32.0, max(width, height) * 1.5))
    return np.asarray([cx, cy, size], dtype=np.float32)


def _default_bbox(width: int, height: int) -> np.ndarray:
    return np.asarray([width * 0.5, height * 0.5, min(width, height) * 0.85], dtype=np.float32)


def _smooth_bboxes(bbx_xys: np.ndarray) -> np.ndarray:
    if len(bbx_xys) <= 1:
        return bbx_xys
    out = bbx_xys.copy()
    alpha = 0.65
    for idx in range(1, len(out)):
        out[idx] = alpha * out[idx - 1] + (1.0 - alpha) * out[idx]
    return out


def _affine_square_crop(frame_rgb: np.ndarray, bbox: np.ndarray, size: int) -> np.ndarray:
    import cv2

    cx, cy, side = [float(v) for v in bbox]
    half = side / 2.0
    src = np.asarray([[cx - half, cy - half], [cx + half, cy - half], [cx, cy]], dtype=np.float32)
    dst = np.asarray([[0, 0], [size - 1, 0], [(size - 1) / 2.0, (size - 1) / 2.0]], dtype=np.float32)
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(frame_rgb, matrix, (size, size), flags=cv2.INTER_LINEAR)


def _vitpose_crops(frames_rgb: np.ndarray, bbx_xys: np.ndarray) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    crops = []
    for frame, bbox in zip(frames_rgb, bbx_xys, strict=True):
        crop256 = _affine_square_crop(frame, bbox, 256)
        crop = crop256[:, 32:224].astype(np.float32) / 255.0
        crops.append(((crop - mean) / std).transpose(2, 0, 1))
    return np.stack(crops).astype(np.float32)


def _hmr2_crops(frames_rgb: np.ndarray, bbx_xys: np.ndarray) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    crops = []
    for frame, bbox in zip(frames_rgb, bbx_xys, strict=True):
        crop = _affine_square_crop(frame, bbox, 256).astype(np.float32) / 255.0
        crops.append(((crop - mean) / std).transpose(2, 0, 1))
    return np.stack(crops).astype(np.float32)


def _batched_trt_infer(runner: Any, input_name: str, values: np.ndarray, output_name: str) -> np.ndarray:
    shape = _binding_shape(runner, input_name)
    batch_size = int(shape[0])
    outputs = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        valid = len(batch)
        if valid < batch_size:
            pad = np.repeat(batch[-1:], batch_size - valid, axis=0)
            batch = np.concatenate([batch, pad], axis=0)
        result = runner.infer({input_name: batch.astype(np.float32)})[output_name]
        outputs.append(result[:valid].copy())
    return np.concatenate(outputs, axis=0)


def _heatmaps_to_keypoints(heatmaps: np.ndarray, bbx_xys: np.ndarray) -> np.ndarray:
    n, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    argmax = flat.argmax(-1)
    px = (argmax % w).astype(np.float32)
    py = (argmax // w).astype(np.float32)
    preds = np.stack([px, py], axis=-1)
    cx = bbx_xys[:, 0:1]
    cy = bbx_xys[:, 1:2]
    side = bbx_xys[:, 2:3]
    preds[..., 0] = preds[..., 0] / w * side * (192.0 / 256.0) + (cx - side * (192.0 / 256.0) / 2)
    preds[..., 1] = preds[..., 1] / h * side + (cy - side / 2)
    conf = 1.0 / (1.0 + np.exp(-flat.max(-1)))
    return np.concatenate([preds, conf[..., None]], axis=-1)


def _static_camera_tensors(length: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    focal = float(max(width, height))
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = float(width) / 2.0
    K[1, 2] = float(height) / 2.0
    K_fullimg = np.repeat(K[None], length, axis=0)
    cam_angvel = np.zeros((length, 6), dtype=np.float32)
    cam_angvel[:, 0] = 1.0
    cam_angvel[:, 4] = 1.0
    return K_fullimg, cam_angvel


def _pose_data_from_decode_outputs(
    *,
    decode_out: dict[str, np.ndarray],
    cam_angvel: np.ndarray | None = None,
    source_fps: int,
    target_fps: int,
    num_frames_to_send: int,
) -> dict[str, np.ndarray]:
    import torch
    from gear_sonic.utils.teleop import genmo_bridge as bridge

    if {"body_pose", "global_orient", "transl"}.issubset(decode_out):
        body_params = {
            "body_pose": torch.as_tensor(np.squeeze(decode_out["body_pose"], axis=0)).float(),
            "global_orient": torch.as_tensor(np.squeeze(decode_out["global_orient"], axis=0)).float(),
            "transl": torch.as_tensor(np.squeeze(decode_out["transl"], axis=0)).float(),
        }
    elif {
        "body_pose_r6d",
        "global_orient_r6d",
        "global_orient_gv_r6d",
        "local_transl_vel",
    }.issubset(decode_out):
        if cam_angvel is None:
            raise ValueError("cam_angvel is required for raw GEM decode outputs")

        body_pose_r6d = torch.as_tensor(decode_out["body_pose_r6d"]).float()
        global_orient_r6d = torch.as_tensor(decode_out["global_orient_r6d"]).float()
        global_orient_gv_r6d = torch.as_tensor(decode_out["global_orient_gv_r6d"]).float()
        local_transl_vel = torch.as_tensor(decode_out["local_transl_vel"]).float()
        cam_angvel_t = torch.as_tensor(cam_angvel[None]).float()
        batch, length = body_pose_r6d.shape[:2]
        body_pose = _matrix_to_axis_angle_torch(
            _rotation_6d_to_matrix_torch(body_pose_r6d.reshape(batch, length, -1, 6))
        ).flatten(-2)
        global_orient_c = _matrix_to_axis_angle_torch(_rotation_6d_to_matrix_torch(global_orient_r6d))
        global_orient_gv = _matrix_to_axis_angle_torch(
            _rotation_6d_to_matrix_torch(global_orient_gv_r6d)
        )
        global_body = _get_body_params_w_Rt_v2_local(
            global_orient_gv=global_orient_gv,
            local_transl_vel=local_transl_vel,
            global_orient_c=global_orient_c,
            cam_angvel=cam_angvel_t,
        )
        body_params = {
            "body_pose": body_pose.squeeze(0),
            "global_orient": global_body["global_orient"].squeeze(0),
            "transl": global_body["transl"].squeeze(0),
        }
    else:
        raise KeyError(
            "Unsupported GEM decode outputs. Expected either body_pose/global_orient/transl "
            "or raw body_pose_r6d/global_orient_r6d/global_orient_gv_r6d/local_transl_vel. "
            f"Got: {sorted(decode_out.keys())}"
        )
    body_pose = bridge._to_time_major_tensor(body_params["body_pose"], 63, "body_pose")
    global_orient = bridge._to_time_major_tensor(body_params["global_orient"], 3, "global_orient")
    transl = bridge._to_time_major_tensor(body_params["transl"], 3, "transl")
    global_orient_quat = bridge.angle_axis_to_quaternion(global_orient)
    global_orient_quat = bridge.smpl_root_ytoz_up(global_orient_quat)
    transl_zup = bridge._rotate_translation_y_to_z_up(transl)
    global_orient_zup = torch.zeros_like(global_orient)
    from scipy.spatial.transform import Rotation as R

    global_orient_zup[:] = torch.as_tensor(
        R.from_quat(global_orient_quat.cpu().numpy()[:, [1, 2, 3, 0]]).as_rotvec(),
        dtype=torch.float32,
    )
    joints_world = bridge.compute_human_joints(
        body_pose=body_pose,
        global_orient=global_orient_zup,
        human_joints_info_path=str(bridge._HUMAN_JOINTS_INFO_PATH),
    )
    body_quat_w = bridge.remove_smpl_base_rot(global_orient_quat, w_last=False)
    root_quat_inv = bridge.quat_inv(body_quat_w).unsqueeze(1).repeat(1, joints_world.shape[1], 1)
    smpl_joints_local = bridge.quat_apply(root_quat_inv, joints_world)
    smpl_pose_np = body_pose.view(-1, 21, 3).cpu().numpy().astype(np.float32)
    target_indices = bridge._resample_num_frames(smpl_pose_np.shape[0], source_fps, target_fps)
    sequence = bridge.BridgeSequence(
        smpl_pose=bridge._resample_axis_angle(smpl_pose_np, target_indices),
        smpl_joints=bridge._resample_linear(smpl_joints_local.cpu().numpy().astype(np.float32), target_indices),
        body_quat_w=bridge._resample_quaternion(body_quat_w.cpu().numpy().astype(np.float32), target_indices),
        joint_pos=bridge.compute_g1_wrist_joint_pos_from_smpl_pose(
            bridge._resample_axis_angle(smpl_pose_np, target_indices)
        ),
        joint_vel=np.zeros((len(target_indices), 29), dtype=np.float32),
        body_pos=bridge._resample_linear(transl_zup.cpu().numpy().astype(np.float32), target_indices),
        source_fps=source_fps,
        target_fps=target_fps,
    )
    # Realtime control should publish the freshest decoded frames in the window.
    # Publishing from 0 adds almost a full-window phase lag and can leave the
    # robot stopped before the video's final pose.
    window_start = max(0, int(sequence.smpl_pose.shape[0]) - int(num_frames_to_send))
    return bridge.build_pose_window(
        sequence=sequence,
        window_start=window_start,
        num_frames_to_send=num_frames_to_send,
        frame_index_start=0,
    )


def _rotation_6d_to_matrix_torch(d6: Any) -> Any:
    import torch
    import torch.nn.functional as F

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _quaternion_to_matrix_torch(quaternions: Any) -> Any:
    import torch

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp_min(1e-8)
    out = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return out.reshape(quaternions.shape[:-1] + (3, 3))


def _axis_angle_to_matrix_torch(axis_angle: Any) -> Any:
    import torch

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    small = angles.abs() < 1e-6
    scale = torch.empty_like(angles)
    scale[~small] = torch.sin(half_angles[~small]) / angles[~small]
    scale[small] = 0.5 - (angles[small] * angles[small]) / 48.0
    quat = torch.cat([torch.cos(half_angles), axis_angle * scale], dim=-1)
    return _quaternion_to_matrix_torch(quat)


def _matrix_to_quaternion_torch(matrix: Any) -> Any:
    import torch

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    denom = 2.0 * q_abs[..., None].clamp_min(0.1)
    quat_candidates = quat_by_rijk / denom
    indices = q_abs.argmax(dim=-1, keepdim=True)
    gather_indices = indices.unsqueeze(-1).expand(batch_dim + (1, 4))
    quat = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def _matrix_to_axis_angle_torch(matrix: Any) -> Any:
    import torch

    quaternions = _matrix_to_quaternion_torch(matrix)
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    scale = 0.5 * torch.sinc(half_angles / torch.pi)
    return quaternions[..., 1:] / scale.clamp_min(1e-8)


def _gaussian_smooth_torch(x: Any, *, sigma: float = 3.0, dim: int = -2) -> Any:
    import torch
    import torch.nn.functional as F

    radius = int(4 * sigma + 0.5)
    grid = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    x_t = x.transpose(dim, -1)
    shape = x_t.shape
    flat = x_t.reshape(-1, 1, shape[-1])
    flat = F.pad(flat, (radius, radius), mode="replicate")
    smoothed = F.conv1d(flat, kernel)
    return smoothed.reshape(shape).transpose(-1, dim)


def _rollout_local_transl_vel_torch(local_transl_vel: Any, global_orient: Any) -> Any:
    import torch

    global_orient_R = _axis_angle_to_matrix_torch(global_orient)
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel)
    transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    return torch.cumsum(torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2), dim=-2)


def _get_body_params_w_Rt_v2_local(
    *,
    global_orient_gv: Any,
    local_transl_vel: Any,
    global_orient_c: Any,
    cam_angvel: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    def as_identity(rot: Any) -> Any:
        is_identity = _matrix_to_axis_angle_torch(rot).norm(dim=-1) < 1e-5
        if is_identity.any():
            eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
            rot = rot.clone()
            rot[is_identity] = eye
        return rot

    batch = cam_angvel.shape[0]
    R_t_to_tp1 = as_identity(_rotation_6d_to_matrix_torch(cam_angvel))
    R_gv = _axis_angle_to_matrix_torch(global_orient_gv)
    R_c = _axis_angle_to_matrix_torch(global_orient_c)
    R_c2gv = R_gv @ R_c.transpose(-1, -2)
    view_axis_gv = R_c2gv[..., 2]
    R_cnext2gv = R_c2gv @ R_t_to_tp1.transpose(-1, -2)
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1, eps=1e-8)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1, eps=1e-8)
    aa_tp1_to_t = torch.cross(vec2_xyz, vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(
        torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0)
    )
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1, eps=1e-8) * aa_tp1_to_t_angle
    aa_tp1_to_t = _gaussian_smooth_torch(aa_tp1_to_t, dim=-2)
    R_tp1_to_t = _axis_angle_to_matrix_torch(aa_tp1_to_t).transpose(-1, -2)

    R_t_to_0 = [torch.eye(3, dtype=R_t_to_tp1.dtype, device=R_t_to_tp1.device)[None].expand(batch, -1, -1)]
    for idx in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, idx])
    R_t_to_0 = as_identity(torch.stack(R_t_to_0, dim=1))

    global_orient = _matrix_to_axis_angle_torch(R_t_to_0 @ R_gv)
    transl = _rollout_local_transl_vel_torch(local_transl_vel, global_orient)
    return {"global_orient": global_orient, "transl": transl}


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0
