# SONIC 视频遥操作复现指南

[English](README.en.md)

本仓库是基于以下两个上游项目的 video teleoperation overlay：

- GR00T-WholeBodyControl: https://github.com/NVlabs/GR00T-WholeBodyControl
- GENMO: https://github.com/NVlabs/GENMO

目标链路：

```text
本地视频
  -> 实时滑动窗口
  -> GENMO TensorRT 推理
  -> SMPL / G1 wrist 参考动作
  -> SONIC ZMQ Protocol v3
  -> MuJoCo 仿真或 G1 真机
```

## Demo

Demo GIF 放在 `assets/demo/`。README 展示用的图片、GIF、截图通常放在
`assets/`、`assets/demo/` 或 `docs/assets/`；`examples/` 更常用于可运行样例。

<p align="center">
  <img src="assets/demo/demo1.gif" width="48%" alt="视频遥操作 demo 1" />
  <img src="assets/demo/demo2.gif" width="48%" alt="视频遥操作 demo 2" />
</p>

<p align="center">
  <strong>真机部署 Demo</strong><br />
  <img src="assets/demo/real_robot.gif" width="72%" alt="真机视频遥操作 demo" />
</p>

## 目录约定

下面所有命令使用 `$WORKSPACE`，请先设置成你的工作目录：

```bash
export WORKSPACE=/path/to/your/workspace
```

推荐目录结构：

```text
$WORKSPACE/
  GENMO/
  GR00T-WholeBodyControl/
  sonic-video-teleop/
  dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4
```

## 1. 克隆上游仓库

```bash
cd "$WORKSPACE"

git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
git clone https://github.com/NVlabs/GENMO.git
```

按两个上游仓库的官方说明完成基础环境安装。本项目当前假设：

```bash
# SONIC / GR00T 环境
conda activate sonic

# GENMO 环境
source "$WORKSPACE/GENMO/.venv/bin/activate"
```

快速检查：

```bash
conda activate sonic
python -c "import cv2, zmq, numpy, scipy; print('sonic env ok')"

"$WORKSPACE/GENMO/.venv/bin/python" -c "import torch; print(torch.__version__)"
```

## 2. 应用本仓库 overlay

```bash
cd "$WORKSPACE/sonic-video-teleop"

rsync -a GR00T-WholeBodyControl/ "$WORKSPACE/GR00T-WholeBodyControl/"
rsync -a GENMO/ "$WORKSPACE/GENMO/"
```

至少应包含：

```text
GR00T-WholeBodyControl/local_video_realtime_bridge/
GR00T-WholeBodyControl/gear_sonic/utils/teleop/genmo_bridge.py
GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config_zmq_smpl_no_root_ori.yaml
GENMO/realtime_trt/
```

确认：

```bash
ls "$WORKSPACE/GR00T-WholeBodyControl/local_video_realtime_bridge/realtime_trt_stream.py"
ls "$WORKSPACE/GENMO/realtime_trt/build_engines.py"
```

## 3. 准备模型和视频

需要以下 GENMO 模型文件：

```text
$WORKSPACE/GENMO/inputs/pretrained/gem_smpl.ckpt
$WORKSPACE/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt
```

检查：

```bash
ls -lh "$WORKSPACE/GENMO/inputs/pretrained/gem_smpl.ckpt"
ls -lh "$WORKSPACE/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt"
```

示例视频路径：

```text
$WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4
```

## 4. 安装 / 检查 TensorRT 依赖

```bash
cd "$WORKSPACE/GENMO"
source .venv/bin/activate

python realtime_trt/diagnose_env.py
python realtime_trt/install_deps.py --dry-run
python realtime_trt/install_deps.py
```

如果需要 ONNX Runtime GPU 对比：

```bash
python realtime_trt/install_deps.py --with-onnxruntime-gpu
```

检查 `trtexec`：

```bash
which trtexec
trtexec --version
```

如果找不到 `trtexec`，请把 TensorRT 的 `bin` 目录加入 `PATH`。

## 5. 构建 TensorRT Engines

```bash
cd "$WORKSPACE/GENMO"
source .venv/bin/activate

python realtime_trt/build_engines.py \
  --example-video "$WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4" \
  --build-engines \
  --fp16 \
  --window-frames 32
```

输出目录：

```text
$WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/engines
```

关键文件：

```text
yolo.engine
vitpose_b8.engine
hmr2_b8.engine
gem_denoiser_explicit_w32.engine
gem_condition_w32.engine
gem_decode_w32.engine
gem_runtime_metadata.json
```

检查：

```bash
cd "$WORKSPACE/GR00T-WholeBodyControl"
ls -lh outputs/realtime_trt/engines
cat outputs/realtime_trt/build_engines_summary.json
```

## 6. Smoke Test

先不连接 ZMQ，只跑 1 个窗口检查 runtime：

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

失败时查看：

```text
GR00T-WholeBodyControl/outputs/realtime_trt/runtime_blocker.json
```

## 7. MuJoCo 仿真运行

终端 1，控制端：

```bash
cd "$WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy"
conda activate sonic

bash deploy.sh \
  --obs-config policy/release/observation_config_zmq_smpl_no_root_ori.yaml \
  --input-type zmq \
  --zmq-host localhost \
  sim
```

终端 2，视频输入和 ZMQ 发布：

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

控制端按键：

```text
]      启动控制
ENTER  开启 ZMQ streaming
O      急停 / 退出
```

## 8. 真机运行

真机前请先用完全相同的视频在 MuJoCo 仿真中验证。

终端 1，真机控制端：

```bash
cd "$WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy"
conda activate sonic

bash deploy.sh \
  --obs-config policy/release/observation_config_zmq_smpl_no_root_ori.yaml \
  --input-type zmq \
  --zmq-host localhost \
  real
```

终端 2 使用与仿真相同的视频发布命令。

真机动作不够平滑时可先尝试：

```bash
--num-frames-to-send 24
--publish-smoothing-alpha 0.75
```

如果源视频 FPS 元数据不准：

```bash
--source-fps-override 30
```

## 9. 不要提交的大文件

本仓库只保存源码和说明文档。不要提交：

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
