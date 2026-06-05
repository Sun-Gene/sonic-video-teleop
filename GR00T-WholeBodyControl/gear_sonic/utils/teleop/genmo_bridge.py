from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

_HUMAN_JOINTS_INFO_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "human" / "human_joints_info.pkl"
)
_Y_TO_Z_UP_QUAT = angle_axis_to_quaternion(torch.tensor([[math.pi / 2, 0.0, 0.0]])).cpu()


@dataclass(slots=True)
class BridgeSequence:
    smpl_pose: np.ndarray
    smpl_joints: np.ndarray
    body_quat_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos: np.ndarray
    source_fps: int
    target_fps: int

    @property
    def num_frames(self) -> int:
        return int(self.smpl_pose.shape[0])


def _to_time_major_tensor(value: object, feature_dim: int, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Missing required field '{name}'")
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.float()
    if tensor.ndim == 1:
        if tensor.shape[0] != feature_dim:
            raise ValueError(f"Field '{name}' must have feature_dim={feature_dim}, got {tensor.shape}")
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim >= 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[-1] * tensor.shape[-2] == feature_dim:
        tensor = tensor.reshape(tensor.shape[0], feature_dim)
    if tensor.ndim != 2 or tensor.shape[-1] != feature_dim:
        raise ValueError(
            f"Field '{name}' must be shaped [T,{feature_dim}] or [1,T,{feature_dim}], got {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


def _load_body_params(smpl_params_path: str | Path) -> dict[str, torch.Tensor]:
    smpl_params = torch.load(smpl_params_path, map_location="cpu")
    if not isinstance(smpl_params, dict):
        raise ValueError(f"Expected dict in {smpl_params_path}, got {type(smpl_params)!r}")

    if "body_params_global" in smpl_params:
        body_params = smpl_params["body_params_global"]
    elif "body_params_incam" in smpl_params:
        body_params = smpl_params["body_params_incam"]
    else:
        body_params = smpl_params

    if not isinstance(body_params, dict):
        raise ValueError("SMPL parameters must be a dict of tensors")
    return body_params


def _quat_lerp_normalized_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    quat = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(quat)
    if norm > 0.0:
        quat = quat / norm
    return quat.astype(np.float32)


def _sanitize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    quats = np.nan_to_num(np.asarray(quaternions, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    invalid = norms.squeeze(-1) < 1e-8
    if np.any(invalid):
        quats[invalid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats / np.clip(norms, 1e-8, None)


def _safe_decompose_rotation_aa(
    rotation_aa: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(rotation_aa, dtype=np.float32)
    twist = np.zeros((rot.shape[0], 4), dtype=np.float32)
    swing = np.zeros((rot.shape[0], 4), dtype=np.float32)
    twist[:, 0] = 1.0
    swing[:, 0] = 1.0

    valid = np.linalg.norm(rot, axis=-1) > 1e-8
    if np.any(valid):
        twist_valid, swing_valid = decompose_rotation_aa(rot[valid], axis)
        twist[valid] = _sanitize_quaternions_wxyz(twist_valid)
        swing[valid] = _sanitize_quaternions_wxyz(swing_valid)
    return twist, swing


def _interp_pose_axis_angle(prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float) -> np.ndarray:
    prev_quats = R.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()
    curr_quats = R.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for idx in range(prev_quats.shape[0]):
        dot = float(np.dot(prev_quats[idx], curr_quats[idx]))
        curr_quat = -curr_quats[idx] if dot < 0.0 else curr_quats[idx]
        out_quat = (1.0 - alpha) * prev_quats[idx] + alpha * curr_quat
        out_quats[idx] = out_quat / np.linalg.norm(out_quat)
    return R.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape).astype(np.float32)


def _resample_linear(values: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    num_frames = values.shape[0]
    if num_frames == 1:
        return np.repeat(values, len(target_indices), axis=0).astype(np.float32)
    left = np.floor(target_indices).astype(np.int64)
    right = np.clip(left + 1, 0, num_frames - 1)
    alpha = (target_indices - left).astype(np.float32)
    alpha_view = alpha.reshape((-1,) + (1,) * (values.ndim - 1))
    out = (1.0 - alpha_view) * values[left] + alpha_view * values[right]
    return out.astype(np.float32)


def _resample_axis_angle(values: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    num_frames = values.shape[0]
    if num_frames == 1:
        return np.repeat(values, len(target_indices), axis=0).astype(np.float32)
    out = np.empty((len(target_indices),) + values.shape[1:], dtype=np.float32)
    for idx, target_index in enumerate(target_indices):
        left = int(math.floor(float(target_index)))
        right = min(left + 1, num_frames - 1)
        alpha = float(target_index - left)
        if left == right or alpha <= 0.0:
            out[idx] = values[left]
        else:
            out[idx] = _interp_pose_axis_angle(values[left], values[right], alpha)
    return out


def _resample_quaternion(values: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    num_frames = values.shape[0]
    if num_frames == 1:
        return np.repeat(values, len(target_indices), axis=0).astype(np.float32)
    out = np.empty((len(target_indices), 4), dtype=np.float32)
    for idx, target_index in enumerate(target_indices):
        left = int(math.floor(float(target_index)))
        right = min(left + 1, num_frames - 1)
        alpha = float(target_index - left)
        if left == right or alpha <= 0.0:
            out[idx] = values[left]
        else:
            out[idx] = _quat_lerp_normalized_wxyz(values[left], values[right], alpha)
    return out


def _resample_num_frames(num_frames: int, source_fps: int, target_fps: int) -> np.ndarray:
    if num_frames <= 1 or source_fps == target_fps:
        return np.arange(num_frames, dtype=np.float32)
    duration = (num_frames - 1) / float(source_fps)
    target_num_frames = max(1, int(round(duration * target_fps)) + 1)
    return np.linspace(0.0, num_frames - 1, target_num_frames, dtype=np.float32)


def _rotate_translation_y_to_z_up(transl: torch.Tensor) -> torch.Tensor:
    base_quat = _Y_TO_Z_UP_QUAT.to(transl)
    return quat_apply(base_quat.repeat(transl.shape[0], 1), transl)


def compute_g1_wrist_joint_pos_from_smpl_pose(smpl_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(smpl_pose, dtype=np.float32)
    squeeze_output = False
    if pose.ndim == 2:
        pose = pose[None, ...]
        squeeze_output = True
    if pose.ndim != 3 or pose.shape[1:] != (21, 3):
        raise ValueError(f"Expected smpl_pose shape [T,21,3] or [21,3], got {pose.shape}")

    joint_pos = np.zeros((pose.shape[0], 29), dtype=np.float32)

    smpl_l_elbow_aa = pose[:, 17]
    smpl_l_wrist_aa = pose[:, 19]
    smpl_r_elbow_aa = pose[:, 18]
    smpl_r_wrist_aa = pose[:, 20]

    g1_l_elbow_q_twist, g1_l_elbow_q_swing = _safe_decompose_rotation_aa(
        smpl_l_elbow_aa, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    g1_r_elbow_q_twist, g1_r_elbow_q_swing = _safe_decompose_rotation_aa(
        smpl_r_elbow_aa, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    _ = g1_l_elbow_q_twist, g1_r_elbow_q_twist

    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    joint_pos[:, 23] = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    joint_pos[:, 25] = l_wrist_euler[:, 1]
    joint_pos[:, 27] = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    joint_pos[:, 24] = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    joint_pos[:, 26] = -r_wrist_euler[:, 1]
    joint_pos[:, 28] = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    return joint_pos[0] if squeeze_output else joint_pos


def load_genmo_bridge_sequence(
    smpl_params_path: str | Path,
    source_fps: int = 30,
    target_fps: int = 50,
) -> BridgeSequence:
    body_params = _load_body_params(smpl_params_path)
    body_pose = _to_time_major_tensor(body_params.get("body_pose"), 63, "body_pose")
    global_orient = _to_time_major_tensor(body_params.get("global_orient"), 3, "global_orient")
    transl = _to_time_major_tensor(body_params.get("transl"), 3, "transl")

    global_orient_quat = angle_axis_to_quaternion(global_orient)
    global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    transl_zup = _rotate_translation_y_to_z_up(transl)
    global_orient_zup = torch.zeros_like(global_orient)
    global_orient_zup[:] = torch.as_tensor(
        R.from_quat(global_orient_quat.cpu().numpy()[:, [1, 2, 3, 0]]).as_rotvec(),
        dtype=torch.float32,
    )

    joints_world = compute_human_joints(
        body_pose=body_pose,
        global_orient=global_orient_zup,
        human_joints_info_path=str(_HUMAN_JOINTS_INFO_PATH),
    )

    body_quat_w = remove_smpl_base_rot(global_orient_quat, w_last=False)
    root_quat_inv = quat_inv(body_quat_w).unsqueeze(1).repeat(1, joints_world.shape[1], 1)
    smpl_joints_local = quat_apply(root_quat_inv, joints_world)

    smpl_pose_np = body_pose.view(-1, 21, 3).cpu().numpy().astype(np.float32)
    smpl_joints_np = smpl_joints_local.cpu().numpy().astype(np.float32)
    body_quat_np = body_quat_w.cpu().numpy().astype(np.float32)
    body_pos_np = transl_zup.cpu().numpy().astype(np.float32)

    target_indices = _resample_num_frames(smpl_pose_np.shape[0], source_fps, target_fps)
    smpl_pose_resampled = _resample_axis_angle(smpl_pose_np, target_indices)
    smpl_joints_resampled = _resample_linear(smpl_joints_np, target_indices)
    body_quat_resampled = _resample_quaternion(body_quat_np, target_indices)
    body_pos_resampled = _resample_linear(body_pos_np, target_indices)
    joint_pos_resampled = compute_g1_wrist_joint_pos_from_smpl_pose(smpl_pose_resampled)
    joint_vel_resampled = np.zeros_like(joint_pos_resampled, dtype=np.float32)

    return BridgeSequence(
        smpl_pose=smpl_pose_resampled,
        smpl_joints=smpl_joints_resampled,
        body_quat_w=body_quat_resampled,
        joint_pos=joint_pos_resampled,
        joint_vel=joint_vel_resampled,
        body_pos=body_pos_resampled,
        source_fps=source_fps,
        target_fps=target_fps,
    )


def build_pose_window(
    sequence: BridgeSequence,
    window_start: int,
    num_frames_to_send: int,
    frame_index_start: int,
) -> dict[str, np.ndarray]:
    if sequence.num_frames == 0:
        raise ValueError("Bridge sequence has no frames")

    max_valid_start = max(sequence.num_frames - num_frames_to_send, 0)
    start = int(np.clip(window_start, 0, max_valid_start))
    stop = min(start + num_frames_to_send, sequence.num_frames)

    smpl_pose = sequence.smpl_pose[start:stop]
    smpl_joints = sequence.smpl_joints[start:stop]
    body_quat_w = sequence.body_quat_w[start:stop]
    joint_pos = sequence.joint_pos[start:stop]
    joint_vel = sequence.joint_vel[start:stop]

    if stop - start < num_frames_to_send:
        pad_count = num_frames_to_send - (stop - start)
        smpl_pose = np.concatenate([smpl_pose, np.repeat(smpl_pose[-1:], pad_count, axis=0)], axis=0)
        smpl_joints = np.concatenate(
            [smpl_joints, np.repeat(smpl_joints[-1:], pad_count, axis=0)], axis=0
        )
        body_quat_w = np.concatenate(
            [body_quat_w, np.repeat(body_quat_w[-1:], pad_count, axis=0)], axis=0
        )
        joint_pos = np.concatenate([joint_pos, np.repeat(joint_pos[-1:], pad_count, axis=0)], axis=0)
        joint_vel = np.concatenate([joint_vel, np.repeat(joint_vel[-1:], pad_count, axis=0)], axis=0)

    frame_index = np.arange(
        frame_index_start,
        frame_index_start + num_frames_to_send,
        dtype=np.int64,
    )

    return {
        "smpl_pose": smpl_pose.astype(np.float32),
        "smpl_joints": smpl_joints.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "frame_index": frame_index,
    }


def publish_bridge_sequence(
    sequence: BridgeSequence,
    zmq_host: str = "*",
    zmq_port: int = 5556,
    zmq_topic: str = "pose",
    num_frames_to_send: int = 5,
    loop: bool = False,
) -> None:
    import zmq

    if num_frames_to_send <= 0:
        raise ValueError("num_frames_to_send must be positive")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    endpoint = f"tcp://{zmq_host}:{zmq_port}"
    socket.bind(endpoint)
    time.sleep(0.1)
    print(f"[genmo_bridge] Publishing {sequence.num_frames} frames on {endpoint} topic='{zmq_topic}'")

    try:
        absolute_frame_index = 0
        max_window_start = max(sequence.num_frames - num_frames_to_send, 0)
        frame_period = 1.0 / float(sequence.target_fps)
        while True:
            for window_start in range(max_window_start + 1):
                payload = build_pose_window(
                    sequence=sequence,
                    window_start=window_start,
                    num_frames_to_send=num_frames_to_send,
                    frame_index_start=absolute_frame_index,
                )
                socket.send(pack_pose_message(payload, topic=zmq_topic, version=3))
                absolute_frame_index += 1
                time.sleep(frame_period)
            if not loop:
                break
    finally:
        socket.close()
        context.term()


def _write_csv(array: np.ndarray, output_path: Path, headers: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat = array.reshape(array.shape[0], -1)
    np.savetxt(
        output_path,
        flat,
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.8f",
    )


def export_motion_directory(sequence: BridgeSequence, output_motion_dir: str | Path) -> Path:
    motion_dir = Path(output_motion_dir)
    motion_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        sequence.joint_pos,
        motion_dir / "joint_pos.csv",
        [f"joint_{idx}" for idx in range(sequence.joint_pos.shape[1])],
    )
    _write_csv(
        sequence.joint_vel,
        motion_dir / "joint_vel.csv",
        [f"joint_vel_{idx}" for idx in range(sequence.joint_vel.shape[1])],
    )
    _write_csv(
        sequence.body_quat_w[:, None, :],
        motion_dir / "body_quat.csv",
        ["body_0_w", "body_0_x", "body_0_y", "body_0_z"],
    )
    _write_csv(
        sequence.body_pos[:, None, :],
        motion_dir / "body_pos.csv",
        ["body_0_x", "body_0_y", "body_0_z"],
    )
    _write_csv(
        sequence.smpl_joints,
        motion_dir / "smpl_joint.csv",
        [f"smpl_joint_{joint_idx}_{axis}" for joint_idx in range(24) for axis in "xyz"],
    )
    _write_csv(
        sequence.smpl_pose,
        motion_dir / "smpl_pose.csv",
        [f"smpl_pose_{joint_idx}_{axis}" for joint_idx in range(21) for axis in "xyz"],
    )

    metadata = (
        f"Metadata for: {motion_dir.name}\n"
        "==============================\n\n"
        "Body part indexes:\n"
        "[0]\n\n"
        f"Total timesteps: {sequence.num_frames}\n"
    )
    (motion_dir / "metadata.txt").write_text(metadata, encoding="utf-8")
    return motion_dir
