#!/usr/bin/env python3
"""Segmented local-video to SONIC Protocol v3 streamer.

This script intentionally lives outside the main gear_sonic package. It is a
thin experiment harness for local-video "online-ish" validation:

  video chunk -> GENMO/GEM -> SMPL bridge sequence -> ZMQ Protocol v3

While one processed chunk is being published, the next chunk is processed in a
background thread. This reduces gaps for local video, but it is still segment
latency rather than true per-frame real time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

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
class SegmentJob:
    index: int
    frames: list[np.ndarray]
    fps: float
    width: int
    height: int
    video_path: Path


@dataclass(slots=True)
class ProcessedSegment:
    index: int
    frames: list[np.ndarray]
    fps: float
    sequence: BridgeSequence
    smpl_params_path: Path


def _default_genmo_python(genmo_repo: Path) -> Path:
    return genmo_repo / ".venv" / "bin" / "python"


def _resolve_path(path: str | None, default: Path) -> Path:
    selected = Path(path).expanduser() if path else default.expanduser()
    if selected.is_absolute():
        return selected
    return Path.cwd() / selected


def _read_segment(
    cap: cv2.VideoCapture,
    segment_index: int,
    frames_per_segment: int,
    fps: float,
    width: int,
    height: int,
    temp_dir: Path,
) -> SegmentJob | None:
    frames: list[np.ndarray] = []
    for _ in range(frames_per_segment):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    if not frames:
        return None

    video_path = temp_dir / f"segment_{segment_index:06d}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open segment writer: {video_path}")

    for frame in frames:
        writer.write(frame)
    writer.release()

    return SegmentJob(
        index=segment_index,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        video_path=video_path,
    )


def _process_segment(
    job: SegmentJob,
    *,
    genmo_repo: Path,
    genmo_python: Path,
    genmo_output_root: Path,
    ckpt_path: Path,
    hmr2_ckpt: Path,
    source_fps: int,
    target_fps: int,
    static_cam: bool,
) -> ProcessedSegment:
    cmd = [
        str(genmo_python),
        "scripts/demo/demo_smpl_hpe.py",
        "--video",
        str(job.video_path),
        "--ckpt_path",
        str(ckpt_path),
        "--hmr2_ckpt",
        str(hmr2_ckpt),
        "--output_root",
        str(genmo_output_root),
        "--no_render",
    ]
    if static_cam:
        cmd.append("--static_cam")

    print(f"[segment {job.index}] GENMO start: {job.video_path.name}", flush=True)
    print(f"[segment {job.index}] GENMO python: {genmo_python}", flush=True)
    started = time.monotonic()
    subprocess.run(cmd, cwd=genmo_repo, check=True)

    smpl_params_path = genmo_output_root / job.video_path.stem / "smpl_params.pt"
    if not smpl_params_path.exists():
        raise FileNotFoundError(f"GENMO did not write expected file: {smpl_params_path}")

    sequence = load_genmo_bridge_sequence(
        smpl_params_path=smpl_params_path,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    elapsed = time.monotonic() - started
    print(
        f"[segment {job.index}] GENMO done in {elapsed:.2f}s; "
        f"{sequence.num_frames} target frames",
        flush=True,
    )
    return ProcessedSegment(
        index=job.index,
        frames=job.frames,
        fps=job.fps,
        sequence=sequence,
        smpl_params_path=smpl_params_path,
    )


def _preview_frame(frame: np.ndarray, scale: float, window_name: str) -> None:
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, frame)


def _publish_segment(
    segment: ProcessedSegment,
    *,
    socket: zmq.Socket,
    topic: str,
    num_frames_to_send: int,
    absolute_frame_index: int,
    no_preview: bool,
    preview_scale: float,
    window_name: str,
) -> int:
    sequence = segment.sequence
    if sequence.num_frames <= 0:
        return absolute_frame_index

    frame_period = 1.0 / float(sequence.target_fps)
    max_window_start = max(sequence.num_frames - num_frames_to_send, 0)
    started = time.monotonic()

    print(
        f"[segment {segment.index}] publish {sequence.num_frames} frames "
        f"from {segment.smpl_params_path}",
        flush=True,
    )

    for target_frame in range(sequence.num_frames):
        window_start = min(target_frame, max_window_start)
        payload = build_pose_window(
            sequence=sequence,
            window_start=window_start,
            num_frames_to_send=num_frames_to_send,
            frame_index_start=absolute_frame_index,
        )
        socket.send(pack_pose_message(payload, topic=topic, version=3))
        absolute_frame_index += 1

        if not no_preview and segment.frames:
            t_video = target_frame / float(sequence.target_fps)
            video_frame_idx = min(int(round(t_video * segment.fps)), len(segment.frames) - 1)
            _preview_frame(segment.frames[video_frame_idx], preview_scale, window_name)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                raise KeyboardInterrupt

        next_time = started + (target_frame + 1) * frame_period
        sleep_s = next_time - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    return absolute_frame_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a local mp4 through chunked GENMO into SONIC ZMQ Protocol v3."
    )
    parser.add_argument("--video", required=True, help="Input mp4 file")
    parser.add_argument(
        "--genmo-repo",
        default="/path/to/workspace/GENMO",
        help="Path to GENMO repository",
    )
    parser.add_argument(
        "--genmo-python",
        default=None,
        help="GENMO python executable; default: <genmo-repo>/.venv/bin/python",
    )
    parser.add_argument(
        "--genmo-output-root",
        default=str(REPO_ROOT / "outputs" / "genmo_realtime_segments"),
        help="Where GENMO writes per-segment artifacts",
    )
    parser.add_argument(
        "--ckpt-path",
        default="/path/to/workspace/GENMO/inputs/pretrained/gem_smpl.ckpt",
        help="GEM-SMPL checkpoint path",
    )
    parser.add_argument(
        "--hmr2-ckpt",
        default="/path/to/workspace/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt",
        help="HMR2 checkpoint path",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=2.0,
        help="Local-video chunk length. Lower reduces chunk latency but increases GENMO overhead.",
    )
    parser.add_argument(
        "--start-second",
        type=float,
        default=0.0,
        help="Start offset in the input video",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Stop after this many segments; 0 means process the whole video.",
    )
    parser.add_argument("--loop-video", action="store_true", help="Loop the input video forever")
    parser.add_argument("--static-cam", action="store_true", help="Pass --static_cam to GENMO")
    parser.add_argument(
        "--target-fps",
        type=int,
        default=50,
        help="SONIC stream FPS",
    )
    parser.add_argument(
        "--source-fps",
        type=int,
        default=0,
        help="GENMO output source FPS. 0 means round input video FPS.",
    )
    parser.add_argument("--zmq-host", default="*", help="Publisher bind host/interface")
    parser.add_argument("--zmq-port", type=int, default=5556, help="Publisher bind port")
    parser.add_argument("--zmq-topic", default="pose", help="ZMQ topic")
    parser.add_argument(
        "--num-frames-to-send",
        type=int,
        default=5,
        help="Sliding future window per ZMQ message",
    )
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV video window")
    parser.add_argument("--preview-scale", type=float, default=0.5, help="Video preview scale")
    parser.add_argument(
        "--keep-temp-video",
        action="store_true",
        help="Keep generated segment mp4 files in the temp directory",
    )
    parser.add_argument(
        "--temp-root",
        default=None,
        help="Optional temp directory root for segment mp4 files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    genmo_repo = Path(args.genmo_repo).expanduser().resolve()
    genmo_python = _resolve_path(args.genmo_python, _default_genmo_python(genmo_repo))
    genmo_output_root = Path(args.genmo_output_root).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    hmr2_ckpt = Path(args.hmr2_ckpt).expanduser().resolve()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")

    if args.start_second > 0.0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start_second * fps)))

    source_fps = args.source_fps if args.source_fps > 0 else max(1, int(round(fps)))
    frames_per_segment = max(1, int(round(args.segment_seconds * fps)))

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{video_path.stem}_segments_",
            dir=args.temp_root,
        )
    )
    print(f"[setup] video={video_path}")
    print(f"[setup] fps={fps:.3f}, size={width}x{height}, source_fps={source_fps}")
    print(f"[setup] segment_seconds={args.segment_seconds}, frames_per_segment={frames_per_segment}")
    print(f"[setup] temp_dir={temp_dir}")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    endpoint = f"tcp://{args.zmq_host}:{args.zmq_port}"
    socket.bind(endpoint)
    time.sleep(0.2)
    print(f"[setup] publishing Protocol v3 on {endpoint} topic='{args.zmq_topic}'")

    window_name = "SONIC local video source"
    if not args.no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def submit_segment(job: SegmentJob | None, executor: ThreadPoolExecutor) -> Future | None:
        if job is None:
            return None
        return executor.submit(
            _process_segment,
            job,
            genmo_repo=genmo_repo,
            genmo_python=genmo_python,
            genmo_output_root=genmo_output_root,
            ckpt_path=ckpt_path,
            hmr2_ckpt=hmr2_ckpt,
            source_fps=source_fps,
            target_fps=args.target_fps,
            static_cam=args.static_cam,
        )

    absolute_frame_index = 0
    segment_index = 0

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first_job = _read_segment(
                cap, segment_index, frames_per_segment, fps, width, height, temp_dir
            )
            current_future = submit_segment(first_job, executor)
            if current_future is None:
                print("[done] no frames to process")
                return
            segment_index += 1

            while current_future is not None:
                current = current_future.result()

                next_job = None
                if args.max_segments <= 0 or segment_index < args.max_segments:
                    next_job = _read_segment(
                        cap, segment_index, frames_per_segment, fps, width, height, temp_dir
                    )
                    if next_job is None and args.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        next_job = _read_segment(
                            cap, segment_index, frames_per_segment, fps, width, height, temp_dir
                        )
                next_future = submit_segment(next_job, executor)
                if next_job is not None:
                    segment_index += 1

                absolute_frame_index = _publish_segment(
                    current,
                    socket=socket,
                    topic=args.zmq_topic,
                    num_frames_to_send=args.num_frames_to_send,
                    absolute_frame_index=absolute_frame_index,
                    no_preview=args.no_preview,
                    preview_scale=args.preview_scale,
                    window_name=window_name,
                )

                current_future = next_future
    except KeyboardInterrupt:
        print("\n[stop] interrupted by user")
    finally:
        cap.release()
        socket.close()
        context.term()
        if not args.no_preview:
            cv2.destroyWindow(window_name)
        if args.keep_temp_video:
            print(f"[cleanup] kept temp videos: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
