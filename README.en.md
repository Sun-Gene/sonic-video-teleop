# SONIC Video Teleoperation Reproduction Guide

[中文](README.zh-CN.md)

This repository is a video teleoperation overlay for:

- GR00T-WholeBodyControl: https://github.com/NVlabs/GR00T-WholeBodyControl
- GENMO: https://github.com/NVlabs/GENMO

Target pipeline:

```text
local video
  -> realtime sliding window
  -> GENMO TensorRT inference
  -> SMPL / G1 wrist reference motion
  -> SONIC ZMQ Protocol v3
  -> MuJoCo simulation or real G1 robot
```

## Demo

Demo GIFs live in `assets/demo/`. For README media, `assets/`,
`assets/demo/`, or `docs/assets/` are common conventions; `examples/` is usually
reserved for runnable samples.

<p align="center">
  <img src="assets/demo/demo1.gif" width="48%" alt="Video teleoperation demo 1" />
  <img src="assets/demo/demo2.gif" width="48%" alt="Video teleoperation demo 2" />
</p>

## Workspace Layout

All commands use `$WORKSPACE`. Set it first:

```bash
export WORKSPACE=/path/to/your/workspace
```

Recommended layout:

```text
$WORKSPACE/
  GENMO/
  GR00T-WholeBodyControl/
  sonic-video-teleop/
  dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4
```

## 1. Clone Upstream Repositories

```bash
cd "$WORKSPACE"

git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
git clone https://github.com/NVlabs/GENMO.git
```

Follow the official setup instructions in both upstream repositories. This
overlay assumes:

```bash
# SONIC / GR00T environment
conda activate sonic

# GENMO environment
source "$WORKSPACE/GENMO/.venv/bin/activate"
```

Sanity checks:

```bash
conda activate sonic
python -c "import cv2, zmq, numpy, scipy; print('sonic env ok')"

"$WORKSPACE/GENMO/.venv/bin/python" -c "import torch; print(torch.__version__)"
```

## 2. Apply This Overlay

```bash
cd "$WORKSPACE/sonic-video-teleop"

rsync -a GR00T-WholeBodyControl/ "$WORKSPACE/GR00T-WholeBodyControl/"
rsync -a GENMO/ "$WORKSPACE/GENMO/"
```

At minimum, the overlay provides:

```text
GR00T-WholeBodyControl/local_video_realtime_bridge/
GR00T-WholeBodyControl/gear_sonic/utils/teleop/genmo_bridge.py
GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config_zmq_smpl_no_root_ori.yaml
GENMO/realtime_trt/
```

Check:

```bash
ls "$WORKSPACE/GR00T-WholeBodyControl/local_video_realtime_bridge/realtime_trt_stream.py"
ls "$WORKSPACE/GENMO/realtime_trt/build_engines.py"
```

## 3. Prepare Models And Video

Required GENMO model files:

```text
$WORKSPACE/GENMO/inputs/pretrained/gem_smpl.ckpt
$WORKSPACE/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt
```

Check:

```bash
ls -lh "$WORKSPACE/GENMO/inputs/pretrained/gem_smpl.ckpt"
ls -lh "$WORKSPACE/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt"
```

Example video used below:

```text
$WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4
```

## 4. Install / Diagnose TensorRT Dependencies

```bash
cd "$WORKSPACE/GENMO"
source .venv/bin/activate

python realtime_trt/diagnose_env.py
python realtime_trt/install_deps.py --dry-run
python realtime_trt/install_deps.py
```

Optional ONNX Runtime GPU comparison:

```bash
python realtime_trt/install_deps.py --with-onnxruntime-gpu
```

Check `trtexec`:

```bash
which trtexec
trtexec --version
```

If `trtexec` is missing, add the TensorRT `bin` directory to `PATH`.

## 5. Build TensorRT Engines

```bash
cd "$WORKSPACE/GENMO"
source .venv/bin/activate

python realtime_trt/build_engines.py \
  --example-video "$WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4" \
  --build-engines \
  --fp16 \
  --window-frames 32
```

Engine output directory:

```text
$WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/engines
```

Key files:

```text
yolo.engine
vitpose_b8.engine
hmr2_b8.engine
gem_denoiser_explicit_w32.engine
gem_condition_w32.engine
gem_decode_w32.engine
gem_runtime_metadata.json
```

Check:

```bash
cd "$WORKSPACE/GR00T-WholeBodyControl"
ls -lh outputs/realtime_trt/engines
cat outputs/realtime_trt/build_engines_summary.json
```

## 6. Smoke Test

Run one window without publishing to ZMQ:

```bash
cd "$WORKSPACE/GR00T-WholeBodyControl"
conda activate sonic

python local_video_realtime_bridge/realtime_trt_stream.py \
  --source video \
  --video "$WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4" \
  --genmo-repo "$WORKSPACE/GENMO" \
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

If it fails, inspect:

```text
GR00T-WholeBodyControl/outputs/realtime_trt/runtime_blocker.json
```

## 7. MuJoCo Simulation

Terminal 1, control side:

```bash
cd "$WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy"
conda activate sonic

bash deploy.sh \
  --obs-config policy/release/observation_config_zmq_smpl_no_root_ori.yaml \
  --input-type zmq \
  --zmq-host localhost \
  sim
```

Terminal 2, video input and ZMQ publisher:

```bash
cd "$WORKSPACE/GR00T-WholeBodyControl"
conda activate sonic

python local_video_realtime_bridge/realtime_trt_stream.py \
  --source video \
  --video "$WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4" \
  --genmo-repo "$WORKSPACE/GENMO" \
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

Control-side keys:

```text
]      start control
ENTER  enable ZMQ streaming
O      emergency stop / exit
```

## 8. Real Robot

Validate the exact same video in MuJoCo before running on hardware.

Terminal 1, real robot control side:

```bash
cd "$WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy"
conda activate sonic

bash deploy.sh \
  --obs-config policy/release/observation_config_zmq_smpl_no_root_ori.yaml \
  --input-type zmq \
  --zmq-host localhost \
  real
```

Terminal 2 uses the same publisher command as the simulation section.

For smoother real-robot motion, try:

```bash
--num-frames-to-send 24
--publish-smoothing-alpha 0.75
```

If the source video FPS metadata is wrong:

```bash
--source-fps-override 30
```

## 9. Files Not Committed

This repository stores source code and documentation only. Do not commit:

```text
*.engine
*.onnx
*.pt
*.pth
*.ckpt
*.mp4
*.avi
*.mov
GR00T-WholeBodyControl/outputs/
GENMO/yolov8x.engine
GENMO/yolov8x.onnx
```
