# Realtime Video Streaming Baseline Snapshot

Date: 2026-05-20

This file records two useful rollback points from the realtime video-to-robot debugging process.

## Why This Snapshot Exists

The rotation problem has gone through two different stages:

1. The robot originally had an approximately 180 degree heading error.
2. After yaw/frame alignment and streaming-window fixes, the error was reduced to approximately 90 degrees.
3. With the latest clean baseline using `num_frames_to_send=16` and yaw-invariant SMPL orientation observations, the robot appears to mostly stop rotating, but now has backward-lean and falling instability.

These are not the same bug. Future fixes for the falling/backward-lean issue should not accidentally erase the earlier working rotation fixes.

## Checkpoint A: 180 Degree To 90 Degree Rotation Baseline

This checkpoint means:

- The large 180 degree coordinate-frame mismatch was reduced.
- The robot could still rotate by about 90 degrees.
- This was useful because it proved the previous issue was mainly a global heading/frame convention problem, not only a low-level controller problem.

Important ingredients:

- `local_video_realtime_bridge/realtime_trt_stream.py`
  - Keep the realtime ZMQ protocol-v3 SMPL streaming path.
  - Keep the neutral/startup gating support.
  - Keep the yaw-offset debugging arguments available, but do not treat a constant yaw offset as the final solution.
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp`
  - Keep streamed-heading reinitialization tied to entering streamed motion, not repeated uncontrolled heading resets on every chunk.
- Runtime command shape:
  - Prefer larger sent chunks, especially `--num-frames-to-send 16`, over very small chunks such as `5`.
  - Keep overlap large enough for TensorRT window continuity.

Typical command:

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
conda activate sonic
python local_video_realtime_bridge/realtime_trt_stream.py \
  --source video \
  --video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --genmo-repo $WORKSPACE/GENMO \
  --engine-root outputs/realtime_trt/engines \
  --engine-window-frames 32 \
  --target-fps 50 \
  --num-frames-to-send 16 \
  --startup-frames 8 \
  --tail-hold-seconds 2.0 \
  --zmq-host '*' \
  --zmq-port 5556 \
  --zmq-topic pose \
  --preview-scale 0.5 \
  --observed-frames 4 \
  --overlap-frames 28 \
  --ddim-steps 8
```

Avoid using these as permanent fixes:

- `--lock-initial-yaw`
- Large constant `--body-yaw-offset-deg`
- Large constant `--root-frame-yaw-offset-deg`
- Large constant `--smpl-joints-yaw-offset-deg`

They are useful for diagnosis, but they can create sliding, leaning, or unstable compensation because they rotate only part of the motion representation.

## Checkpoint B: Current No-Rotation Baseline

This checkpoint means:

- The robot appears to mostly stop rotating.
- The remaining problem is backward-lean/falling stability.
- This is the current best baseline for fixing stability.

Important additional ingredient:

- `gear_sonic_deploy/policy/release/observation_config.yaml`
  - SMPL mode uses:

```yaml
smpl_anchor_orientation_refheading_10frame_step1
```

instead of:

```yaml
smpl_anchor_orientation_10frame_step1
```

Reason:

- The full SMPL anchor orientation contains global heading/yaw.
- The refheading variant reduces sensitivity to global yaw, so the policy is less tempted to turn the whole robot to match a possibly inconsistent realtime heading.

Known remaining issue:

- Removing or reducing the global heading cue can expose posture-distribution problems.
- The policy may now receive a yaw-stable target, but the SMPL joints/root posture can still put the implied center of mass behind the support polygon.
- Therefore the robot may lean backward and fall even though it no longer spins.

## Files To Protect

Current rotation-related files:

- `local_video_realtime_bridge/realtime_trt_stream.py`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp`
- `gear_sonic_deploy/policy/release/observation_config.yaml`
- `local_video_realtime_bridge/realtime_video_to_robot_rotation_analysis.md`

Before changing stability logic, compare against this snapshot and confirm whether the experiment is based on Checkpoint A or Checkpoint B.

## How To Recognize Regression

Likely rotation regression:

- First streamed `body_quat` has a yaw near +/-90 degrees or flips sign across windows.
- Robot turns in place soon after switching to `Reference motion name: streamed`.
- Changing `--root-frame-yaw-offset-deg` between `90` and `-90` changes direction but does not fix stability.

Likely stability regression:

- Robot no longer turns much, but torso pitches backward.
- Feet slide while the body tries to follow future SMPL targets.
- Falling happens after the neutral/startup section when dynamic future frames enter the window.

## Next Debugging Target

For the current no-rotation baseline, do not start by adding another constant yaw offset.

Debug in this order:

1. Compare offline and realtime SMPL root-local joints for the first 30 to 50 frames.
2. Check whether pelvis, torso, ankle, and foot points imply a backward center-of-mass shift.
3. Print or log the actual SMPL observation tensors used by the policy, especially:
   - `smpl_joints_10frame_step1`
   - `smpl_anchor_orientation_refheading_10frame_step1`
   - `motion_joint_positions_wrists_10frame_step1`
4. Add a short startup blend/ramp from the robot standing pose to streamed SMPL targets.
5. If needed, implement a yawless orientation feature that preserves roll/pitch but removes only yaw, instead of relying on broad constant rotation offsets.
