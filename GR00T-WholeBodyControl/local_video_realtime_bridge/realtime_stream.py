#!/usr/bin/env python3
"""Realtime-ish video/camera to SONIC Protocol v3 streamer.

This is an isolated experimental harness inspired by the SONIC paper's video
teleoperation stack. It keeps the source preview/capture loop independent from
GENMO inference and ZMQ publishing:

  frame source -> sliding windows -> persistent GENMO worker -> Protocol v3

The open-source PyTorch GENMO path is still much slower than the paper's
TensorRT runtime, so this script uses latest-data-wins queues instead of
blocking the source video/camera when inference falls behind.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import cv2
import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.genmo_bridge import (  # noqa: E402
    BridgeSequence,
    build_pose_window,
    load_genmo_bridge_sequence,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402


@dataclass(slots=True)
class WindowJob:
    job_id: int
    window_index: int
    video_path: Path
    output_dir: Path
    source_fps: int
    frame_count: int
    capture_started_at: float
    capture_ended_at: float
    overlap_seconds: float


@dataclass(slots=True)
class ProcessedWindow:
    job: WindowJob
    sequence: BridgeSequence
    smpl_params_path: Path
    worker_elapsed_s: float
    bridge_elapsed_s: float
    stage_times: dict[str, float]


class LatestQueue:
    """A tiny latest-data-wins queue.

    New puts evict any queued old item. The item currently being processed by a
    consumer is not interrupted; consumers can poll for newer items between
    frames to switch early.
    """

    def __init__(self, maxsize: int = 1) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)

    def put_latest(self, item: Any) -> list[Any]:
        dropped: list[Any] = []
        while True:
            try:
                dropped.append(self._queue.get_nowait())
            except queue.Empty:
                break
        self._queue.put(item)
        return dropped

    def get(self, timeout: float | None = None) -> Any:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> Any:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()


def _sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "session"


def _resolve_path(path: str | None, default: Path) -> Path:
    selected = Path(path).expanduser() if path else default.expanduser()
    if selected.is_absolute():
        return selected
    return (Path.cwd() / selected).resolve()


def _default_genmo_python(genmo_repo: Path) -> Path:
    return genmo_repo / ".venv" / "bin" / "python"


def _draw_preview(frame: np.ndarray, lines: list[str], scale: float) -> np.ndarray:
    out = frame
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    y = 24
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 80), 1, cv2.LINE_AA)
        y += 24
    return out


def _resize_for_processing(frame: np.ndarray, process_width: int, process_height: int) -> np.ndarray:
    if process_width <= 0 and process_height <= 0:
        return frame
    h, w = frame.shape[:2]
    if process_width > 0 and process_height > 0:
        size = (process_width, process_height)
    elif process_width > 0:
        scale = process_width / float(w)
        size = (process_width, max(1, int(round(h * scale))))
    else:
        scale = process_height / float(h)
        size = (max(1, int(round(w * scale))), process_height)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _write_window_video(
    frames: list[np.ndarray],
    path: Path,
    fps: float,
    process_width: int,
    process_height: int,
) -> None:
    if not frames:
        raise ValueError("Cannot write an empty window")
    path.parent.mkdir(parents=True, exist_ok=True)
    first = _resize_for_processing(frames[0], process_width, process_height)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        writer.write(first)
        for frame in frames[1:]:
            writer.write(_resize_for_processing(frame, process_width, process_height))
    finally:
        writer.release()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


class GenmoWorkerClient:
    def __init__(
        self,
        *,
        genmo_python: Path,
        genmo_repo: Path,
        ckpt_path: Path,
        hmr2_ckpt: Path,
        static_cam: bool,
        write_overlay: bool,
        result_queue: LatestQueue,
        target_fps: int,
        keep_window_videos: bool,
    ) -> None:
        self.genmo_python = genmo_python
        self.genmo_repo = genmo_repo
        self.ckpt_path = ckpt_path
        self.hmr2_ckpt = hmr2_ckpt
        self.static_cam = static_cam
        self.write_overlay = write_overlay
        self.result_queue = result_queue
        self.target_fps = target_fps
        self.keep_window_videos = keep_window_videos

        self.pending = LatestQueue(maxsize=1)
        self.proc: subprocess.Popen[str] | None = None
        self.runner_thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.idle_cond = threading.Condition()
        self.in_flight = False

    def start(self) -> None:
        worker_path = Path(__file__).with_name("genmo_worker.py")
        cmd = [
            str(self.genmo_python),
            str(worker_path),
            "--genmo-repo",
            str(self.genmo_repo),
            "--ckpt-path",
            str(self.ckpt_path),
            "--hmr2-ckpt",
            str(self.hmr2_ckpt),
        ]
        if self.static_cam:
            cmd.append("--static-cam")
        if self.write_overlay:
            cmd.append("--write-overlay")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
            env=env,
        )
        self.stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self.stderr_thread.start()

        ready = self._read_json_line()
        if not ready.get("ok") or ready.get("type") != "ready":
            raise RuntimeError(f"GENMO worker failed to start: {ready}")
        print("[realtime] GENMO worker ready")

        self.runner_thread = threading.Thread(target=self._runner, daemon=True)
        self.runner_thread.start()

    def submit_latest(self, job: WindowJob) -> None:
        dropped = self.pending.put_latest(job)
        for old_job in dropped:
            print(f"[capture] drop queued window {old_job.window_index} (latest-data-wins)")
            if not self.keep_window_videos:
                _safe_unlink(old_job.video_path)
        with self.idle_cond:
            self.idle_cond.notify_all()

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self.idle_cond:
            while self.in_flight or not self.pending.empty():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.idle_cond.wait(timeout=remaining)
            return True

    def close(self, terminate: bool = False) -> None:
        self.stop_event.set()
        if self.runner_thread is not None:
            self.pending.put_latest(None)
            self.runner_thread.join(timeout=5.0)
        if self.proc is not None:
            if terminate and self.proc.poll() is None:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5.0)

    def _pump_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            print(line.rstrip(), flush=True)

    def _read_json_line(self) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("GENMO worker exited before returning JSON")
        return json.loads(line)

    def _runner(self) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        try:
            while not self.stop_event.is_set():
                try:
                    job = self.pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                if job is None:
                    break

                with self.idle_cond:
                    self.in_flight = True
                    self.idle_cond.notify_all()

                request = {
                    "type": "process",
                    "job_id": job.job_id,
                    "video_path": str(job.video_path),
                    "output_dir": str(job.output_dir),
                }
                self.proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                self.proc.stdin.flush()

                response = self._read_json_line()
                if not response.get("ok"):
                    print(f"[worker] job {job.job_id} failed: {response.get('error')}")
                    print(response.get("traceback", ""))
                    with self.idle_cond:
                        self.in_flight = False
                        self.idle_cond.notify_all()
                    continue

                bridge_start = time.monotonic()
                sequence = load_genmo_bridge_sequence(
                    smpl_params_path=response["smpl_params_path"],
                    source_fps=job.source_fps,
                    target_fps=self.target_fps,
                )
                bridge_elapsed = time.monotonic() - bridge_start
                processed = ProcessedWindow(
                    job=job,
                    sequence=sequence,
                    smpl_params_path=Path(response["smpl_params_path"]),
                    worker_elapsed_s=float(response["elapsed_s"]),
                    bridge_elapsed_s=bridge_elapsed,
                    stage_times={k: float(v) for k, v in response.get("stage_times", {}).items()},
                )
                dropped = self.result_queue.put_latest(processed)
                for old_result in dropped:
                    print(f"[publish] drop processed window {old_result.job.window_index}")
                    if not self.keep_window_videos:
                        _safe_unlink(old_result.job.video_path)
                print(
                    f"[worker] window={job.window_index} worker={processed.worker_elapsed_s:.2f}s "
                    f"bridge={bridge_elapsed:.2f}s target_frames={sequence.num_frames}"
                )

                with self.idle_cond:
                    self.in_flight = False
                    self.idle_cond.notify_all()

            try:
                self.proc.stdin.write('{"type":"shutdown"}\n')
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        finally:
            with self.idle_cond:
                self.in_flight = False
                self.idle_cond.notify_all()


class PosePublisher:
    def __init__(
        self,
        *,
        result_queue: LatestQueue,
        zmq_host: str,
        zmq_port: int,
        zmq_topic: str,
        target_fps: int,
        num_frames_to_send: int,
        dry_run_zmq: bool,
        no_publish: bool,
        preview_mode: str,
        preview_scale: float,
        keep_window_videos: bool,
    ) -> None:
        self.result_queue = result_queue
        self.zmq_host = zmq_host
        self.zmq_port = zmq_port
        self.zmq_topic = zmq_topic
        self.target_fps = target_fps
        self.num_frames_to_send = num_frames_to_send
        self.dry_run_zmq = dry_run_zmq
        self.no_publish = no_publish or dry_run_zmq
        self.preview_mode = preview_mode
        self.preview_scale = preview_scale
        self.keep_window_videos = keep_window_videos
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.absolute_frame_index = 0
        self.has_published_any = False
        self._idle_lock = threading.Lock()
        self._busy = False

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._idle_lock:
                if not self._busy and self.result_queue.empty():
                    return True
            time.sleep(0.05)
        return False

    def _set_busy(self, busy: bool) -> None:
        with self._idle_lock:
            self._busy = busy

    def _run(self) -> None:
        context: zmq.Context | None = None
        socket: zmq.Socket | None = None
        if not self.no_publish:
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            endpoint = f"tcp://{self.zmq_host}:{self.zmq_port}"
            socket.bind(endpoint)
            time.sleep(0.2)
            print(f"[publish] Protocol v3 on {endpoint} topic='{self.zmq_topic}'")
        else:
            print("[publish] no-publish/dry-run mode: payloads are built but not sent")

        frame_period = 1.0 / float(self.target_fps)
        current: ProcessedWindow | None = None
        try:
            while not self.stop_event.is_set() or not self.result_queue.empty() or current is not None:
                if current is None:
                    try:
                        current = self.result_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                self._set_busy(True)
                sequence = current.sequence
                publish_start = 0 if not self.has_published_any else int(
                    round(current.job.overlap_seconds * self.target_fps)
                )
                publish_start = min(max(0, publish_start), max(sequence.num_frames - 1, 0))
                lag_s = time.monotonic() - current.job.capture_ended_at
                print(
                    f"[publish] window={current.job.window_index} frames={sequence.num_frames} "
                    f"start={publish_start} lag={lag_s:.2f}s"
                )

                target_frame = publish_start
                interrupted = False
                started = time.monotonic()
                preview_cap: cv2.VideoCapture | None = None
                if self.preview_mode in {"publish", "both"}:
                    preview_cap = cv2.VideoCapture(str(current.job.video_path))
                    if not preview_cap.isOpened():
                        print(f"[publish] warning: cannot open preview video {current.job.video_path}")
                        preview_cap = None
                while target_frame < sequence.num_frames:
                    try:
                        newer = self.result_queue.get_nowait()
                        print(
                            f"[publish] switch window {current.job.window_index} -> "
                            f"{newer.job.window_index} (latest-data-wins)"
                        )
                        current = newer
                        interrupted = True
                        break
                    except queue.Empty:
                        pass

                    if preview_cap is not None:
                        source_frame_idx = int(
                            round(target_frame / float(self.target_fps) * current.job.source_fps)
                        )
                        preview_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, source_frame_idx))
                        ok, frame = preview_cap.read()
                        if ok:
                            preview = _draw_preview(
                                frame,
                                [
                                    f"published window={current.job.window_index}",
                                    f"lag={lag_s:.2f}s frame={target_frame}/{sequence.num_frames}",
                                    "q/ESC stop",
                                ],
                                self.preview_scale,
                            )
                            cv2.imshow("SONIC published video source", preview)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (27, ord("q")):
                                print("[publish] stopped by preview key")
                                self.stop_event.set()
                                break

                    payload = build_pose_window(
                        sequence=sequence,
                        window_start=target_frame,
                        num_frames_to_send=self.num_frames_to_send,
                        frame_index_start=self.absolute_frame_index,
                    )
                    if self.dry_run_zmq and self.absolute_frame_index == 0:
                        summary = {k: list(v.shape) for k, v in payload.items()}
                        print(f"[dry-run] first Protocol v3 payload shapes: {summary}")
                    if socket is not None:
                        socket.send(pack_pose_message(payload, topic=self.zmq_topic, version=3))
                    self.absolute_frame_index += 1
                    self.has_published_any = True
                    target_frame += 1

                    next_time = started + (target_frame - publish_start) * frame_period
                    sleep_s = next_time - time.monotonic()
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)

                if not interrupted:
                    if preview_cap is not None:
                        preview_cap.release()
                    if not self.keep_window_videos:
                        _safe_unlink(current.job.video_path)
                    current = None
                    self._set_busy(False)
                elif preview_cap is not None:
                    preview_cap.release()
        finally:
            self._set_busy(False)
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()
            if self.preview_mode in {"publish", "both"}:
                cv2.destroyWindow("SONIC published video source")


def _open_source(args: argparse.Namespace) -> tuple[cv2.VideoCapture, float, int, int, str, bool]:
    if args.source == "video":
        if not args.video:
            raise ValueError("--video is required when --source video")
        cap = cv2.VideoCapture(str(Path(args.video).expanduser()))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")
        if args.start_second > 0.0:
            fps_probe = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start_second * fps_probe)))
        label = Path(args.video).expanduser().stem
        is_video_file = True
    else:
        if args.gstreamer_pipeline:
            cap = cv2.VideoCapture(args.gstreamer_pipeline, cv2.CAP_GSTREAMER)
            label = "gstreamer"
        elif args.camera_url:
            cap = cv2.VideoCapture(args.camera_url)
            label = "rtsp"
        else:
            cap = cv2.VideoCapture(args.camera_index)
            label = f"camera{args.camera_index}"
        if args.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        if not cap.isOpened():
            raise RuntimeError(
                "Failed to open camera source. Check --camera-index, --camera-url, "
                "or --gstreamer-pipeline."
            )
        is_video_file = False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = float(args.camera_fps if args.source == "camera" and args.camera_fps > 0 else 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return cap, fps, width, height, label, is_video_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime-ish mp4/camera -> persistent GENMO -> SONIC Protocol v3."
    )
    parser.add_argument("--source", choices=("video", "camera"), default="video")
    parser.add_argument("--video", help="Input mp4 path for --source video")
    parser.add_argument("--start-second", type=float, default=0.0)
    parser.add_argument("--loop-video", action="store_true")

    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", default=None, help="RTSP/IP camera URL")
    parser.add_argument("--gstreamer-pipeline", default=None, help="OpenCV GStreamer pipeline")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)

    parser.add_argument("--genmo-repo", default="/path/to/workspace/GENMO")
    parser.add_argument("--genmo-python", default=None)
    parser.add_argument(
        "--ckpt-path",
        default="/path/to/workspace/GENMO/inputs/pretrained/gem_smpl.ckpt",
    )
    parser.add_argument(
        "--hmr2-ckpt",
        default="/path/to/workspace/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt",
    )
    parser.add_argument("--static-cam", action="store_true")
    parser.add_argument("--write-overlay", action="store_true")

    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.5)
    parser.add_argument("--target-fps", type=int, default=50)
    parser.add_argument("--source-fps", type=int, default=0, help="0 means round source FPS")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--process-width", type=int, default=0, help="Resize GENMO input width")
    parser.add_argument("--process-height", type=int, default=0, help="Resize GENMO input height")

    parser.add_argument("--zmq-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq-topic", default="pose")
    parser.add_argument("--num-frames-to-send", type=int, default=5)
    parser.add_argument("--dry-run-zmq", action="store_true")
    parser.add_argument("--no-publish", action="store_true")

    parser.add_argument("--preview-scale", type=float, default=0.5)
    parser.add_argument(
        "--preview-mode",
        choices=("capture", "publish", "both", "none"),
        default="publish",
        help=(
            "capture shows the live input stream; publish shows the delayed "
            "source frames synchronized with ZMQ publishing; both shows both"
        ),
    )
    parser.add_argument("--no-preview", action="store_true", help="Alias for --preview-mode none")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / "realtime_bridge_sessions"),
    )
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--keep-window-videos", action="store_true")
    parser.add_argument("--drain-timeout", type=float, default=300.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.window_seconds <= 0.0:
        raise ValueError("--window-seconds must be positive")
    if args.overlap_seconds < 0.0 or args.overlap_seconds >= args.window_seconds:
        raise ValueError("--overlap-seconds must be >= 0 and < --window-seconds")

    genmo_repo = Path(args.genmo_repo).expanduser().resolve()
    genmo_python = _resolve_path(args.genmo_python, _default_genmo_python(genmo_repo))
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    hmr2_ckpt = Path(args.hmr2_ckpt).expanduser().resolve()

    cap, fps, width, height, source_label, is_video_file = _open_source(args)
    source_fps = args.source_fps if args.source_fps > 0 else max(1, int(round(fps)))
    window_frames = max(1, int(round(args.window_seconds * fps)))
    overlap_frames = max(0, int(round(args.overlap_seconds * fps)))
    hop_frames = max(1, window_frames - overlap_frames)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_name = args.session_name or f"{_sanitize_name(source_label)}_{timestamp}"
    session_root = Path(args.output_root).expanduser().resolve() / session_name
    windows_dir = session_root / "window_videos"
    genmo_root = session_root / "genmo_windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    genmo_root.mkdir(parents=True, exist_ok=True)

    print(f"[setup] source={args.source} label={source_label}")
    print(f"[setup] fps={fps:.3f} rounded_source_fps={source_fps} size={width}x{height}")
    print(
        f"[setup] window={args.window_seconds:.2f}s/{window_frames}f "
        f"overlap={args.overlap_seconds:.2f}s/{overlap_frames}f hop={hop_frames}f"
    )
    print(f"[setup] session={session_root}")

    effective_preview_mode = "none" if args.no_preview else args.preview_mode
    result_queue = LatestQueue(maxsize=1)
    publisher = PosePublisher(
        result_queue=result_queue,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        zmq_topic=args.zmq_topic,
        target_fps=args.target_fps,
        num_frames_to_send=args.num_frames_to_send,
        dry_run_zmq=args.dry_run_zmq,
        no_publish=args.no_publish,
        preview_mode=effective_preview_mode,
        preview_scale=args.preview_scale,
        keep_window_videos=args.keep_window_videos,
    )
    worker = GenmoWorkerClient(
        genmo_python=genmo_python,
        genmo_repo=genmo_repo,
        ckpt_path=ckpt_path,
        hmr2_ckpt=hmr2_ckpt,
        static_cam=args.static_cam,
        write_overlay=args.write_overlay,
        result_queue=result_queue,
        target_fps=args.target_fps,
        keep_window_videos=args.keep_window_videos,
    )

    worker_started = False
    publisher_started = False
    window_name = "SONIC realtime source"

    try:
        worker_started = True
        worker.start()
        publisher.start()
        publisher_started = True
        if effective_preview_mode in {"capture", "both"}:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frames_buffer: list[np.ndarray] = []
        times_buffer: list[float] = []
        buffer_start_idx = 0
        absolute_source_idx = 0
        next_window_start = 0
        window_index = 0
        source_started = time.monotonic()
        next_video_time = source_started
        stop_capture = False

        while True:
            ok, frame = cap.read()
            if not ok:
                if is_video_file and args.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[capture] source ended")
                break

            now = time.monotonic()
            frames_buffer.append(frame.copy())
            times_buffer.append(now)

            while absolute_source_idx >= next_window_start + window_frames - 1:
                offset = next_window_start - buffer_start_idx
                window_frames_list = [f.copy() for f in frames_buffer[offset : offset + window_frames]]
                window_times = times_buffer[offset : offset + window_frames]
                if len(window_frames_list) != window_frames:
                    break

                window_video = windows_dir / f"window_{window_index:06d}.mp4"
                output_dir = genmo_root / f"window_{window_index:06d}"
                _write_window_video(
                    window_frames_list,
                    window_video,
                    fps=float(source_fps),
                    process_width=args.process_width,
                    process_height=args.process_height,
                )
                job = WindowJob(
                    job_id=window_index,
                    window_index=window_index,
                    video_path=window_video,
                    output_dir=output_dir,
                    source_fps=source_fps,
                    frame_count=window_frames,
                    capture_started_at=window_times[0],
                    capture_ended_at=window_times[-1],
                    overlap_seconds=args.overlap_seconds,
                )
                print(f"[capture] enqueue window={window_index} video={window_video.name}")
                worker.submit_latest(job)

                window_index += 1
                next_window_start += hop_frames
                remove_count = next_window_start - buffer_start_idx
                if remove_count > 0:
                    del frames_buffer[:remove_count]
                    del times_buffer[:remove_count]
                    buffer_start_idx = next_window_start
                if args.max_windows > 0 and window_index >= args.max_windows:
                    print(f"[capture] reached --max-windows {args.max_windows}")
                    stop_capture = True
                    break

            if stop_capture:
                break

            if effective_preview_mode in {"capture", "both"}:
                preview = _draw_preview(
                    frame,
                    [
                        f"{args.source} {source_label}",
                        f"src_frame={absolute_source_idx} windows={window_index}",
                        "q/ESC stop",
                    ],
                    args.preview_scale,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("[capture] stopped by preview key")
                    break

            absolute_source_idx += 1
            if is_video_file:
                next_video_time += 1.0 / fps
                sleep_s = next_video_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

        print("[drain] waiting for pending worker/publisher work")
        if not worker.wait_idle(args.drain_timeout):
            print("[drain] worker still busy after timeout")
        if not publisher.wait_idle(args.drain_timeout):
            print("[drain] publisher still busy after timeout")
    except KeyboardInterrupt:
        print("\n[stop] interrupted")
    finally:
        cap.release()
        if effective_preview_mode in {"capture", "both"}:
            cv2.destroyWindow(window_name)
        if worker_started:
            worker.close()
        if publisher_started:
            publisher.stop()
        print(f"[done] session artifacts: {session_root}")


if __name__ == "__main__":
    main()
