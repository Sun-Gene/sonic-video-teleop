# Video Teleoperation Reproduction Guide

This guide describes how to reproduce the local-video-to-SONIC teleoperation
pipeline used in this workspace.

The pipeline is:

```text
local mp4 / camera frames
  -> realtime sliding window
  -> GENMO TensorRT runtime
  -> SMPL + G1 wrist reference motion
  -> SONIC ZMQ Protocol v3
  -> MuJoCo sim or real G1 robot
```

The upstream repositories are:

- GR00T-WholeBodyControl: https://github.com/NVlabs/GR00T-WholeBodyControl
- GENMO: https://github.com/NVlabs/GENMO

This repository/overlay only contains the video teleoperation changes. Large
generated artifacts such as TensorRT engines, ONNX files, checkpoints, videos,
and runtime windows should not be committed to Git.

## 1. Directory Layout

The instructions below assume this layout:

```text
$WORKSPACE/
  GENMO/
  GR00T-WholeBodyControl/
  dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4
```

If your workspace is elsewhere, replace `$WORKSPACE` in all commands.

## 2. Clone Upstream Repositories

```bash
cd $WORKSPACE

git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
git clone https://github.com/NVlabs/GENMO.git
```

Follow the official installation instructions in the two upstream repositories
to create the base environments. In this workspace:

- GR00T/SONIC runs in the conda environment `sonic`.
- GENMO has a Python virtual environment at `$WORKSPACE/GENMO/.venv`.

Sanity checks:

```bash
conda activate sonic
python -c "import cv2, zmq, numpy, scipy; print('sonic env ok')"

$WORKSPACE/GENMO/.venv/bin/python -c "import torch; print(torch.__version__)"
```

## 3. Apply The Video Teleoperation Overlay

Copy the overlay files into the cloned upstream repositories. If your private
repo stores the same relative paths as this guide, use:

```bash
cd $WORKSPACE/your-private-video-teleop-repo

rsync -a GR00T-WholeBodyControl/ $WORKSPACE/GR00T-WholeBodyControl/
rsync -a GENMO/ $WORKSPACE/GENMO/
```

At minimum, the overlay should provide:

```text
GR00T-WholeBodyControl/local_video_realtime_bridge/
GR00T-WholeBodyControl/gear_sonic/utils/teleop/genmo_bridge.py
GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config_zmq_smpl_no_root_ori.yaml
GENMO/realtime_trt/
```

Optional but useful files:

```text
GR00T-WholeBodyControl/docs/source/tutorials/genmo_video_teleop.md
GR00T-WholeBodyControl/docs/source/tutorials/genmo_video_to_sonic_v3.md
GR00T-WholeBodyControl/gear_sonic/scripts/genmo_to_sonic.py
GR00T-WholeBodyControl/gear_sonic/scripts/video_to_sonic.py
```

Do not copy generated caches:

```text
__pycache__/
*.pyc
*.engine
*.onnx
*.pt
outputs/
```

## 4. Check Required Model Files

The runtime expects these GENMO-side model files:

```text
$WORKSPACE/GENMO/inputs/pretrained/gem_smpl.ckpt
$WORKSPACE/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt
```

Check them:

```bash
ls -lh $WORKSPACE/GENMO/inputs/pretrained/gem_smpl.ckpt
ls -lh $WORKSPACE/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt
```

Also prepare an example video. The commands below use:

```text
$WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4
```

## 5. Install / Diagnose TensorRT Runtime Dependencies

Run diagnostics in the GENMO environment:

```bash
cd $WORKSPACE/GENMO
source .venv/bin/activate

python realtime_trt/diagnose_env.py
python realtime_trt/install_deps.py --dry-run
```

If the dry run looks correct, install runtime dependencies:

```bash
python realtime_trt/install_deps.py
```

Optional, for ONNX Runtime GPU comparisons:

```bash
python realtime_trt/install_deps.py --with-onnxruntime-gpu
```

The build scripts use `trtexec`. Check that TensorRT is visible:

```bash
which trtexec
trtexec --version
```

If `trtexec` is not found, add the TensorRT `bin` directory to `PATH`.

## 6. Build TensorRT Engines

Build all required engines from the GENMO repo:

```bash
cd $WORKSPACE/GENMO
source .venv/bin/activate

python realtime_trt/build_engines.py \
  --example-video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --build-engines \
  --fp16 \
  --window-frames 32
```

Expected output root:

```text
$WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt
```

The runtime reads engines from:

```text
$WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/engines
```

Expected important files:

```text
outputs/realtime_trt/engines/yolo.engine
outputs/realtime_trt/engines/vitpose_b8.engine
outputs/realtime_trt/engines/hmr2_b8.engine
outputs/realtime_trt/engines/gem_denoiser_explicit_w32.engine
outputs/realtime_trt/engines/gem_condition_w32.engine
outputs/realtime_trt/engines/gem_decode_w32.engine
outputs/realtime_trt/engines/gem_runtime_metadata.json
```

Check:

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
ls -lh outputs/realtime_trt/engines
cat outputs/realtime_trt/build_engines_summary.json
```

If only one engine is missing, rebuild only that part with the relevant skip
flags. Example: rebuild only the GEM runtime engines:

```bash
cd $WORKSPACE/GENMO
source .venv/bin/activate

python realtime_trt/build_engines.py \
  --example-video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --skip-yolo \
  --skip-vitpose \
  --skip-hmr2 \
  --build-engines \
  --fp16 \
  --window-frames 32
```

## 7. Optional Runtime Smoke Test

Before connecting to sim or real robot, run one or two windows without ZMQ:

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
  --observed-frames 4 \
  --overlap-frames 28 \
  --ddim-steps 8 \
  --max-windows 2 \
  --no-publish \
  --no-preview
```

If the runtime cannot start, it writes a blocker report:

```text
outputs/realtime_trt/runtime_blocker.json
```

## 8. Run In MuJoCo Sim

Open two terminals.

### Terminal 1: Control Side

```bash
cd $WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy
conda activate sonic

bash deploy.sh \
  --obs-config policy/release/observation_config_zmq_smpl_no_root_ori.yaml \
  --input-type zmq \
  --zmq-host localhost \
  sim
```

After startup, use the keyboard in this terminal:

```text
]      start control
ENTER  enable ZMQ streaming
O      emergency stop / exit
```

### Terminal 2: Video Input And ZMQ Publisher

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

The publisher opens a preview window by default. Add `--no-preview` when running
headless.

## 9. Run On Real Robot

Always validate the exact same video or camera source in sim first. Keep a hand
near the emergency stop key.

Open two terminals.

### Terminal 1: Real Robot Control Side

```bash
cd $WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy
conda activate sonic

bash deploy.sh \
  --obs-config policy/release/observation_config_zmq_smpl_no_root_ori.yaml \
  --input-type zmq \
  --zmq-host localhost \
  real
```

Keyboard:

```text
]      start control
ENTER  enable ZMQ streaming
O      emergency stop / exit
```

### Terminal 2: Video Input And ZMQ Publisher

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

## 10. Useful Runtime Tuning

The default reproduction command keeps the current behavior. For smoother real
robot motion, try a larger ZMQ future window:

```bash
--num-frames-to-send 24
```

If motion still has chunk-to-chunk jitter on the real robot, enable optional
publish smoothing:

```bash
--publish-smoothing-alpha 0.75
```

If an input video has incorrect FPS metadata, override the detected source FPS:

```bash
--source-fps-override 30
```

Lower latency experiments:

```bash
--ddim-steps 6
--no-preview
```

The main latency/status lines are printed by `realtime_trt_stream.py`:

```text
[runtime] inference_done ...
[runtime] status submitted=... completed=... published=... dropped=... last_total=... last_latency=...
[publish] latest-data-wins ...
```

If `dropped` increases quickly or `last_latency` is larger than the available
future buffer, the real robot can pause while waiting for the next streamed
motion chunk.

## 11. What Not To Commit

Keep the private Git repository small. Do not commit:

```text
GR00T-WholeBodyControl/outputs/
GENMO/yolov8x.engine
GENMO/yolov8x.onnx
*.engine
*.onnx
*.pt
*.pth
*.ckpt
*.mp4
*.avi
*.mov
__pycache__/
*.pyc
```

Recommended `.gitignore`:

```gitignore
__pycache__/
*.pyc
*.engine
*.onnx
*.pt
*.pth
*.ckpt
*.mp4
*.avi
*.mov
outputs/
debug/
.env
.venv/
```

## 12. Quick Checklist

```bash
# 1. Engines exist
ls $WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/engines

# 2. Runtime imports
cd $WORKSPACE/GR00T-WholeBodyControl
conda activate sonic
python -m py_compile local_video_realtime_bridge/realtime_trt_stream.py

# 3. Smoke test without publishing
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
  --observed-frames 4 \
  --overlap-frames 28 \
  --ddim-steps 8 \
  --max-windows 1 \
  --no-publish \
  --no-preview
```
