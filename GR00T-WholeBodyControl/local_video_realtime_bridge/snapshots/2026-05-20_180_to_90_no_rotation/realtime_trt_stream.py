#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import queue
from pathlib import Path
import sys
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENMO_ROOT = REPO_ROOT.parent / "GENMO"
DEFAULT_ENGINE_ROOT = REPO_ROOT / "outputs" / "realtime_trt" / "engines"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "realtime_trt" / "runtime_blocker.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _add_genmo_runtime(genmo_repo: Path) -> None:
    genmo_repo = genmo_repo.expanduser().resolve()
    if str(genmo_repo) not in sys.path:
        sys.path.insert(0, str(genmo_repo))


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    import json

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _open_capture(args: argparse.Namespace):
    import cv2

    if args.source == "video":
        if not args.video:
            raise ValueError("--video is required when --source video")
        cap = cv2.VideoCapture(args.video)
    elif args.camera_url:
        cap = cv2.VideoCapture(args.camera_url)
    elif args.gstreamer_pipeline:
        cap = cv2.VideoCapture(args.gstreamer_pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.camera_index)
        if args.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input source")
    return cap


def _draw(frame: np.ndarray, lines: list[str], scale: float) -> np.ndarray:
    import cv2

    out = frame
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    y = 26
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 255, 80), 1, cv2.LINE_AA)
        y += 26
    return out


class LatestSlot:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Any | None = None
        self._closed = False
        self.dropped = 0

    def put_latest(self, item: Any) -> None:
        with self._cond:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._cond.notify_all()

    def get(self, timeout: float = 0.1) -> Any | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._item is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._item is None:
                return None
            item = self._item
            self._item = None
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_result: dict[str, Any] | None = None
        self.latest_error: BaseException | None = None
        self.submitted = 0
        self.completed = 0
        self.published = 0
        self.last_latency_s = 0.0
        self.last_total_s = 0.0

    def set_result(self, result: dict[str, Any]) -> None:
        with self.lock:
            self.latest_result = result
            self.completed += 1
            self.last_latency_s = float(result.get("latency_s", 0.0))
            self.last_total_s = float(result.get("stage_times", {}).get("total_s", 0.0))

    def set_error(self, exc: BaseException) -> None:
        with self.lock:
            self.latest_error = exc

    def mark_submitted(self) -> int:
        with self.lock:
            self.submitted += 1
            return self.submitted

    def mark_published(self) -> int:
        with self.lock:
            self.published += 1
            return self.published

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "latest_result": self.latest_result,
                "latest_error": self.latest_error,
                "submitted": self.submitted,
                "completed": self.completed,
                "published": self.published,
                "last_latency_s": self.last_latency_s,
                "last_total_s": self.last_total_s,
            }


class InitialYawLock:
    def __init__(self, *, enabled: bool, target_yaw_deg: float) -> None:
        self.enabled = enabled
        self.target_yaw_rad = float(target_yaw_deg) * 3.141592653589793 / 180.0
        self._yaw_correction_rad: float | None = None

    def apply(self, pose_data: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return pose_data
        body_quat = pose_data.get("body_quat_w")
        if body_quat is None:
            return pose_data

        import numpy as np
        from scipy.spatial.transform import Rotation as R

        body_quat_np = np.asarray(body_quat, dtype=np.float32)
        if body_quat_np.shape[-1] != 4 or body_quat_np.size == 0:
            return pose_data

        flat = body_quat_np.reshape(-1, 4)
        norms = np.linalg.norm(flat, axis=1)
        valid = np.where(norms > 1e-6)[0]
        if valid.size == 0:
            return pose_data

        if self._yaw_correction_rad is None:
            first_wxyz = flat[int(valid[0])] / max(float(norms[int(valid[0])]), 1e-6)
            first_rot = R.from_quat(first_wxyz[[1, 2, 3, 0]])
            first_yaw = float(first_rot.as_euler("xyz", degrees=False)[2])
            self._yaw_correction_rad = self.target_yaw_rad - first_yaw

        correction = R.from_euler("z", float(self._yaw_correction_rad))
        xyzw = flat[:, [1, 2, 3, 0]]
        rotated = (correction * R.from_quat(xyzw)).as_quat()
        out = rotated[:, [3, 0, 1, 2]].reshape(body_quat_np.shape).astype(np.float32)
        pose_data["body_quat_w"] = out

        for key in ("smpl_joints", "body_pos"):
            value = pose_data.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape[-1] != 3 or arr.size == 0:
                continue
            flat_xyz = arr.reshape(-1, 3)
            pose_data[key] = correction.apply(flat_xyz).reshape(arr.shape).astype(np.float32)

        return pose_data


class BodyYawOffset:
    def __init__(self, *, yaw_offset_deg: float) -> None:
        self.yaw_offset_rad = float(yaw_offset_deg) * 3.141592653589793 / 180.0

    def apply(self, pose_data: dict[str, Any]) -> dict[str, Any]:
        if abs(self.yaw_offset_rad) < 1e-9:
            return pose_data
        body_quat = pose_data.get("body_quat_w")
        if body_quat is None:
            return pose_data

        import numpy as np
        from scipy.spatial.transform import Rotation as R

        body_quat_np = np.asarray(body_quat, dtype=np.float32)
        if body_quat_np.shape[-1] != 4 or body_quat_np.size == 0:
            return pose_data

        flat = body_quat_np.reshape(-1, 4)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-6
        if not np.any(valid):
            return pose_data

        flat_norm = flat.copy()
        flat_norm[valid] = flat_norm[valid] / norms[valid]
        correction = R.from_euler("z", float(self.yaw_offset_rad))
        xyzw = flat_norm[:, [1, 2, 3, 0]]
        rotated = (correction * R.from_quat(xyzw)).as_quat()
        out = rotated[:, [3, 0, 1, 2]].reshape(body_quat_np.shape).astype(np.float32)
        pose_data["body_quat_w"] = out
        return pose_data


class RootFrameYawOffset:
    def __init__(self, *, yaw_offset_deg: float) -> None:
        self.yaw_offset_rad = float(yaw_offset_deg) * 3.141592653589793 / 180.0

    def apply(self, pose_data: dict[str, Any]) -> dict[str, Any]:
        if abs(self.yaw_offset_rad) < 1e-9:
            return pose_data

        import numpy as np
        from scipy.spatial.transform import Rotation as R

        correction = R.from_euler("z", float(self.yaw_offset_rad))

        body_quat = pose_data.get("body_quat_w")
        if body_quat is not None:
            body_quat_np = np.asarray(body_quat, dtype=np.float32)
            if body_quat_np.shape[-1] == 4 and body_quat_np.size:
                flat = body_quat_np.reshape(-1, 4)
                norms = np.linalg.norm(flat, axis=1, keepdims=True)
                valid = norms[:, 0] > 1e-6
                flat_norm = flat.copy()
                flat_norm[valid] = flat_norm[valid] / norms[valid]
                xyzw = flat_norm[:, [1, 2, 3, 0]]
                rotated = (correction * R.from_quat(xyzw)).as_quat()
                pose_data["body_quat_w"] = rotated[:, [3, 0, 1, 2]].reshape(body_quat_np.shape).astype(np.float32)

        # smpl_joints are root-local observations. If the root frame is yaw-shifted
        # in world space, the same joint coordinates must be expressed in the
        # inverse-shifted root-local frame.
        smpl_joints = pose_data.get("smpl_joints")
        if smpl_joints is not None:
            smpl_joints_np = np.asarray(smpl_joints, dtype=np.float32)
            if smpl_joints_np.shape[-1] == 3 and smpl_joints_np.size:
                inv_correction = correction.inv()
                flat_xyz = smpl_joints_np.reshape(-1, 3)
                pose_data["smpl_joints"] = inv_correction.apply(flat_xyz).reshape(smpl_joints_np.shape).astype(np.float32)

        return pose_data


class SmplJointsYawOffset:
    def __init__(self, *, yaw_offset_deg: float) -> None:
        self.yaw_offset_rad = float(yaw_offset_deg) * 3.141592653589793 / 180.0

    def apply(self, pose_data: dict[str, Any]) -> dict[str, Any]:
        if abs(self.yaw_offset_rad) < 1e-9:
            return pose_data
        smpl_joints = pose_data.get("smpl_joints")
        if smpl_joints is None:
            return pose_data

        import numpy as np
        from scipy.spatial.transform import Rotation as R

        smpl_joints_np = np.asarray(smpl_joints, dtype=np.float32)
        if smpl_joints_np.shape[-1] != 3 or smpl_joints_np.size == 0:
            return pose_data

        correction = R.from_euler("z", float(self.yaw_offset_rad))
        flat_xyz = smpl_joints_np.reshape(-1, 3)
        pose_data["smpl_joints"] = correction.apply(flat_xyz).reshape(smpl_joints_np.shape).astype(np.float32)
        return pose_data


class NeutralStartGate:
    def __init__(self, *, enabled: bool, smpl_pose_threshold: float) -> None:
        self.enabled = bool(enabled)
        self.smpl_pose_threshold = float(smpl_pose_threshold)
        self.open = not self.enabled

    def should_publish(self, pose_data: dict[str, Any]) -> bool:
        if self.open:
            return True
        smpl_pose = pose_data.get("smpl_pose")
        if smpl_pose is None:
            self.open = True
            return True

        import numpy as np

        smpl_pose_np = np.asarray(smpl_pose, dtype=np.float32)
        if smpl_pose_np.size == 0:
            return False
        max_abs = float(np.max(np.abs(smpl_pose_np)))
        if max_abs >= self.smpl_pose_threshold:
            self.open = True
            body_quat = pose_data.get("body_quat_w")
            if body_quat is not None:
                body_quat_np = np.asarray(body_quat, dtype=np.float32).reshape(-1, 4)
                if body_quat_np.size:
                    print(
                        "[publish] neutral-start gate opened "
                        f"max_abs_smpl_pose={max_abs:.6f} "
                        f"first_body_quat_wxyz={body_quat_np[0].tolist()}",
                        flush=True,
                    )
                else:
                    print(f"[publish] neutral-start gate opened max_abs_smpl_pose={max_abs:.6f}", flush=True)
            else:
                print(f"[publish] neutral-start gate opened max_abs_smpl_pose={max_abs:.6f}", flush=True)
            return True
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Video/camera to SONIC Protocol v3 stream. The entry is threaded for "
            "latest-data-wins publishing; the runtime mode printed at startup says "
            "whether the loaded GENMO backend is paper_trt or still hybrid."
        )
    )
    parser.add_argument("--source", choices=["video", "camera"], required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", default=None)
    parser.add_argument("--gstreamer-pipeline", default=None)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--genmo-repo", default=str(DEFAULT_GENMO_ROOT))
    parser.add_argument("--genmo-python", default=str(DEFAULT_GENMO_ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--ckpt-path", default=str(DEFAULT_GENMO_ROOT / "inputs" / "pretrained" / "gem_smpl.ckpt"))
    parser.add_argument(
        "--hmr2-ckpt",
        default=str(DEFAULT_GENMO_ROOT / "inputs" / "checkpoints" / "hmr2" / "epoch=10-step=25000-001.ckpt"),
    )
    parser.add_argument("--engine-root", default=str(DEFAULT_ENGINE_ROOT))
    parser.add_argument(
        "--runtime-output-root",
        default=str(REPO_ROOT / "outputs" / "realtime_trt_runtime_windows"),
    )
    parser.add_argument("--engine-window-frames", type=int, default=32)
    parser.add_argument("--observed-frames", type=int, default=8)
    parser.add_argument("--overlap-frames", type=int, default=8)
    parser.add_argument("--target-fps", type=int, default=50)
    parser.add_argument("--num-frames-to-send", type=int, default=5)
    parser.add_argument("--latency-budget-ms", type=float, default=150.0)
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=16,
        help="Number of GEM DDIM denoising steps for realtime TRT. Use 50 for full-quality debug.",
    )
    parser.add_argument(
        "--startup-frames",
        type=int,
        default=8,
        help="Submit the first padded window after this many frames to reduce startup delay.",
    )
    parser.add_argument(
        "--tail-hold-seconds",
        type=float,
        default=2.0,
        help="After video EOF/max-windows, keep waiting/publishing so the final inferred window is not truncated.",
    )
    parser.add_argument("--zmq-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq-topic", default="pose")
    parser.add_argument("--preview-scale", type=float, default=0.5)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument(
        "--lock-initial-yaw",
        action="store_true",
        help="Experimental: map the first generated root yaw to --initial-yaw-deg before publishing.",
    )
    parser.add_argument(
        "--no-lock-initial-yaw",
        action="store_true",
        help="Compatibility no-op; initial yaw locking is disabled unless --lock-initial-yaw is set.",
    )
    parser.add_argument(
        "--initial-yaw-deg",
        type=float,
        default=0.0,
        help="Target world yaw for the first generated body_quat_w frame. Try 180 if the sim faces the opposite way.",
    )
    parser.add_argument(
        "--body-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Apply a constant yaw offset to body_quat_w only. Does not rotate smpl_joints/body_pos.",
    )
    parser.add_argument(
        "--root-frame-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Apply a root-frame yaw convention offset: rotate body_quat_w by +yaw and smpl_joints by -yaw.",
    )
    parser.add_argument(
        "--smpl-joints-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Apply a constant yaw offset to smpl_joints only. Leaves body_quat_w unchanged.",
    )
    parser.add_argument(
        "--skip-neutral-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold ZMQ publishing until the generated SMPL pose is non-neutral at startup.",
    )
    parser.add_argument(
        "--neutral-start-threshold",
        type=float,
        default=1e-4,
        help="Max abs SMPL pose threshold used by --skip-neutral-start.",
    )
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--sync-inference", action="store_true", help="Debug only: block capture while infer_window runs.")
    parser.add_argument("--fallback-hybrid", action="store_true", help="Debug only: allow the old GENMO worker/mp4 path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser


def main() -> int:
    args = build_parser().parse_args()

    genmo_repo = Path(args.genmo_repo).expanduser().resolve()
    _add_genmo_runtime(genmo_repo)

    try:
        from realtime_trt.runtime import EnginePaths, MissingEngineError, RealtimeTRTPipeline
    except Exception as exc:
        report = {
            "ok": False,
            "stage": "import_runtime",
            "error": repr(exc),
            "hint": "Run this from GR00T root and ensure GENMO/realtime_trt exists.",
        }
        output = _write_json(args.report, report)
        print(f"[blocker] wrote {output}")
        return 2

    engine_paths = EnginePaths.from_root(args.engine_root)
    try:
        pipeline = RealtimeTRTPipeline(
            engine_paths,
            window_frames=args.engine_window_frames,
            observed_frames=args.observed_frames,
            overlap_frames=args.overlap_frames,
            latency_budget_ms=args.latency_budget_ms,
            genmo_repo=genmo_repo,
            genmo_python=args.genmo_python,
            ckpt_path=args.ckpt_path,
            hmr2_ckpt=args.hmr2_ckpt,
            output_root=args.runtime_output_root,
            target_fps=args.target_fps,
            num_frames_to_send=args.num_frames_to_send,
            ddim_steps=args.ddim_steps,
            fallback_hybrid=args.fallback_hybrid,
        )
    except MissingEngineError as exc:
        report = {
            "ok": False,
            "stage": "engine_validation",
            "blocker": str(exc),
            "engines": engine_paths.as_dict(),
            "required_next_command": (
                "cd /path/to/workspace/GENMO && source .venv/bin/activate && "
                "python realtime_trt/build_engines.py --example-video /path/to/input.mp4 "
                "--build-engines --fp16 --window-frames 32"
            ),
        }
        output = _write_json(args.report, report)
        print(f"[blocker] Missing TensorRT engines. Details: {output}")
        return 2

    readiness = pipeline.readiness_report()
    if not readiness.get("runtime_bound", False):
        report = {
            "ok": False,
            "stage": "runtime_readiness",
            "blocker": readiness["reason"],
            "engines": engine_paths.as_dict(),
            "missing_runtime_components": readiness["missing_runtime_components"],
            "note": (
                "Engine build success only proves the model fragments can be serialized by TensorRT. "
                "It does not mean the streaming video-to-SMPL-to-Protocol-v3 runtime is wired."
            ),
        }
        output = _write_json(args.report, report)
        print(f"[blocker] Runtime binding incomplete. Details: {output}")
        return 2
    print(f"[runtime] mode={readiness.get('mode', 'unknown')} explicit_gem={readiness.get('gem_denoiser_explicit')}")

    import cv2
    import numpy as np
    endpoint = f"tcp://{args.zmq_host}:{args.zmq_port}"
    context = None
    socket = None
    pack_pose_message = None
    if not args.no_publish:
        import zmq

        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message as _pack_pose_message

        pack_pose_message = _pack_pose_message
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.bind(endpoint)
        time.sleep(0.1)
        print(f"[publish] Protocol v3 on {endpoint} topic='{args.zmq_topic}'")
    else:
        print("[publish] disabled (--no-publish)")

    cap = _open_capture(args)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 1:
        source_fps = args.camera_fps if args.source == "camera" else 30.0
    pipeline.source_fps = int(round(source_fps))
    frame_period = 1.0 / float(source_fps)

    job_slot = LatestSlot()
    shared = SharedState()
    stop_event = threading.Event()
    report: dict[str, Any] = {"ok": False, "stage": "runtime"}
    yaw_lock = InitialYawLock(
        enabled=bool(args.lock_initial_yaw and not args.no_lock_initial_yaw),
        target_yaw_deg=args.initial_yaw_deg,
    )
    body_yaw_offset = BodyYawOffset(yaw_offset_deg=args.body_yaw_offset_deg)
    root_frame_yaw_offset = RootFrameYawOffset(yaw_offset_deg=args.root_frame_yaw_offset_deg)
    smpl_joints_yaw_offset = SmplJointsYawOffset(yaw_offset_deg=args.smpl_joints_yaw_offset_deg)
    neutral_start_gate = NeutralStartGate(
        enabled=bool(args.skip_neutral_start),
        smpl_pose_threshold=args.neutral_start_threshold,
    )

    def run_inference() -> None:
        while not stop_event.is_set():
            job = job_slot.get(timeout=0.1)
            if job is None:
                continue
            try:
                result = pipeline.infer_window(job["frames"], job["capture_ts"])
            except BaseException as exc:  # keep the capture loop from hanging silently
                shared.set_error(exc)
                stop_event.set()
                return
            result["window_id"] = job["window_id"]
            shared.set_result(result)
            stage_times = result.get("stage_times", {})
            total_s = float(stage_times.get("total_s", 0.0))
            print(
                "[runtime] inference_done "
                f"runtime={result.get('mode')} window={job['window_id']} "
                f"total={total_s:.2f}s latency={float(result.get('latency_s', 0.0)):.2f}s "
                f"stages=yolo:{float(stage_times.get('yolo_s', 0.0)):.3f},"
                f"vitpose:{float(stage_times.get('vitpose_s', 0.0)):.3f},"
                f"hmr2:{float(stage_times.get('hmr2_s', 0.0)):.3f},"
                f"cond:{float(stage_times.get('condition_s', 0.0)):.3f},"
                f"ddim:{float(stage_times.get('ddim_s', 0.0)):.3f},"
                f"decode:{float(stage_times.get('decode_s', 0.0)):.3f} "
                f"obs={int(stage_times.get('observed_frames', 0))} "
                f"smpl={result.get('smpl_params_path')}",
                flush=True,
            )

    def run_publisher() -> None:
        if socket is None or pack_pose_message is None:
            return
        import zmq

        send_frame_index = 0
        send_period = 1.0 / float(args.target_fps)
        next_send = time.monotonic()
        last_window_id: int | None = None
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(0.002, next_send - now))
                continue
            next_send += send_period
            snapshot = shared.snapshot()
            result = snapshot["latest_result"]
            if result is None:
                continue
            pose_data = yaw_lock.apply(copy.deepcopy(result["pose_data"]))
            pose_data = root_frame_yaw_offset.apply(pose_data)
            pose_data = smpl_joints_yaw_offset.apply(pose_data)
            pose_data = body_yaw_offset.apply(pose_data)
            if not neutral_start_gate.should_publish(pose_data):
                continue
            pose_data["frame_index"] = np.arange(
                send_frame_index,
                send_frame_index + args.num_frames_to_send,
                dtype=np.int64,
            )
            try:
                socket.send(
                    pack_pose_message(pose_data, topic=args.zmq_topic, version=3),
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                continue
            send_frame_index += 1
            shared.mark_published()
            window_id = int(result.get("window_id", -1))
            if window_id != last_window_id:
                last_window_id = window_id
                print(
                    "[publish] latest-data-wins "
                    f"window={window_id} frame_index={send_frame_index} "
                    f"latency={float(result.get('latency_s', 0.0)):.2f}s",
                    flush=True,
                )

    inference_thread: threading.Thread | None = None
    publisher_thread: threading.Thread | None = None
    if not args.sync_inference:
        inference_thread = threading.Thread(target=run_inference, name="trt-inference", daemon=True)
        inference_thread.start()
        publisher_thread = threading.Thread(target=run_publisher, name="zmq-publisher", daemon=True)
        publisher_thread.start()

    window: list[np.ndarray] = []
    submitted_windows = 0
    sync_frame_index = 0
    last_read = time.monotonic()
    last_status = time.monotonic()
    input_exhausted = False

    try:
        while not stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok:
                input_exhausted = True
                break
            capture_ts = time.monotonic()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            window.append(frame_rgb)
            if len(window) > args.engine_window_frames:
                window = window[-args.engine_window_frames :]

            snapshot = shared.snapshot()
            latest_error = snapshot["latest_error"]
            if latest_error is not None:
                raise latest_error

            if not args.no_preview:
                cv2.imshow(
                    "SONIC TensorRT source",
                    _draw(
                        frame_bgr,
                        [
                            f"source={args.source} fps={source_fps:.1f}",
                            f"runtime={readiness.get('mode', 'unknown')}",
                            f"window={len(window)}/{args.engine_window_frames}",
                            (
                                f"submitted={snapshot['submitted']} completed={snapshot['completed']} "
                                f"published={snapshot['published']} dropped={job_slot.dropped}"
                            ),
                            (
                                f"last_total={snapshot['last_total_s']:.2f}s "
                                f"last_latency={snapshot['last_latency_s']:.2f}s"
                            ),
                        ],
                        args.preview_scale,
                    ),
                )
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            startup_ready = submitted_windows == 0 and len(window) >= max(1, args.startup_frames)
            full_window_ready = len(window) == args.engine_window_frames
            if full_window_ready or startup_ready:
                frames = np.stack(window, axis=0)
                if args.sync_inference:
                    try:
                        result = pipeline.infer_window(frames, capture_ts)
                    except NotImplementedError as exc:
                        report.update(
                            {
                                "blocker": str(exc),
                                "engines": engine_paths.as_dict(),
                                "note": (
                                    "Engine files exist, but binding the captured GENMO denoiser, scheduler, "
                                    "and SMPL postprocess still has to be completed."
                                ),
                            }
                        )
                        output = _write_json(args.report, report)
                        print(f"[blocker] Runtime binding incomplete. Details: {output}")
                        return 2
                    result["window_id"] = submitted_windows
                    shared.set_result(result)
                    if socket is not None and pack_pose_message is not None:
                        import zmq

                        pose_data = yaw_lock.apply(copy.deepcopy(result["pose_data"]))
                        pose_data = root_frame_yaw_offset.apply(pose_data)
                        pose_data = smpl_joints_yaw_offset.apply(pose_data)
                        pose_data = body_yaw_offset.apply(pose_data)
                        pose_data["frame_index"] = np.arange(
                            sync_frame_index,
                            sync_frame_index + args.num_frames_to_send,
                            dtype=np.int64,
                        )
                        socket.send(
                            pack_pose_message(pose_data, topic=args.zmq_topic, version=3),
                            flags=zmq.NOBLOCK,
                        )
                        sync_frame_index += 1
                        shared.mark_published()
                else:
                    job_slot.put_latest(
                        {
                            "window_id": submitted_windows,
                            "frames": frames,
                            "capture_ts": capture_ts,
                        }
                    )
                    shared.mark_submitted()

                submitted_windows += 1
                window = window[-args.overlap_frames :] if args.overlap_frames > 0 else []
                if args.max_windows > 0 and submitted_windows >= args.max_windows:
                    break

            now = time.monotonic()
            if now - last_status >= 1.0:
                snapshot = shared.snapshot()
                print(
                    "[runtime] status "
                    f"runtime={readiness.get('mode', 'unknown')} submitted={snapshot['submitted']} "
                    f"completed={snapshot['completed']} published={snapshot['published']} "
                    f"dropped={job_slot.dropped} last_total={snapshot['last_total_s']:.2f}s "
                    f"last_latency={snapshot['last_latency_s']:.2f}s",
                    flush=True,
                )
                last_status = now

            if args.source == "video":
                elapsed = time.monotonic() - last_read
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)
                last_read = time.monotonic()

        if (
            input_exhausted
            and not args.sync_inference
            and len(window) > max(args.overlap_frames, 0)
            and (args.max_windows <= 0 or submitted_windows < args.max_windows)
        ):
            job_slot.put_latest(
                {
                    "window_id": submitted_windows,
                    "frames": np.stack(window, axis=0),
                    "capture_ts": time.monotonic(),
                }
            )
            shared.mark_submitted()
            submitted_windows += 1

        if submitted_windows > 0 and not args.sync_inference:
            target_window = submitted_windows - 1
            deadline = time.monotonic() + max(0.0, args.tail_hold_seconds) + 2.0
            final_seen_at: float | None = None
            while time.monotonic() < deadline:
                snapshot = shared.snapshot()
                if snapshot["latest_error"] is not None:
                    raise snapshot["latest_error"]
                latest = snapshot["latest_result"]
                if latest is not None and int(latest.get("window_id", -1)) >= target_window:
                    if final_seen_at is None:
                        final_seen_at = time.monotonic()
                    hold_s = 0.0 if args.no_publish else max(0.0, args.tail_hold_seconds)
                    if time.monotonic() - final_seen_at >= hold_s:
                        break
                time.sleep(0.05)
    except NotImplementedError as exc:
        report.update(
            {
                "blocker": str(exc),
                "engines": engine_paths.as_dict(),
                "note": (
                    "Engine files exist, but binding the captured GENMO denoiser, scheduler, "
                    "and SMPL postprocess still has to be completed."
                ),
            }
        )
        output = _write_json(args.report, report)
        print(f"[blocker] Runtime binding incomplete. Details: {output}")
        return 2
    finally:
        stop_event.set()
        job_slot.close()
        if inference_thread is not None:
            inference_thread.join(timeout=2.0)
        if publisher_thread is not None:
            publisher_thread.join(timeout=2.0)
        cap.release()
        pipeline.close()
        if socket is not None:
            socket.close()
        if context is not None:
            context.term()
        if not args.no_preview:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
