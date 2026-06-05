# SONIC 论文级视频遥操作 TensorRT 复现说明

本文档说明 SONIC 论文里视频遥操作为什么可以低延迟实时运行，以及当前公开 GENMO/SONIC 代码要复现到论文级实时需要补齐哪些工程部分。

## 1. TensorRT 在这里的作用

TensorRT 不是一个新模型，而是 NVIDIA GPU 上的推理编译器和 runtime。它通常把 PyTorch/ONNX 模型转换成 `.engine` 文件，在固定或半固定输入形状下做以下优化：

- FP16/INT8 低精度推理，减少显存带宽和算力开销。
- layer fusion，把多个算子合并成更少的 CUDA kernel。
- kernel autotuning，为当前显卡选择更快的实现。
- 静态 shape 优化，减少 PyTorch eager mode 的 Python 调度开销。
- CUDA stream / CUDA graph 友好，适合稳定周期控制环。

对你当前任务来说，TensorRT 的价值不是“让 ZMQ 更快”。ZMQ 传 Protocol v3 的开销很小，主要瓶颈在视频人体观测和 GENMO/GEM 推理：

```text
YOLO bbox -> ViTPose 2D keypoints -> HMR2 image features -> GENMO/GEM diffusion -> SMPL -> Protocol v3
```

之前的 `realtime_stream.py` 虽然让 GEM 模型常驻，但每个 2 秒窗口仍然要做 YOLO、ViTPose、HMR2 和 PyTorch diffusion。你的日志里一个窗口约 16-17 秒，所以视频早就播完，机器人只能很晚收到窗口结果。这不是通信问题，而是推理路径完全没有达到实时。

## 2. SONIC 论文视频遥操作的核心思路

从论文描述看，系统不是用公开 demo 那种“先完整处理一个视频片段，再播放动作”的方式。它更像下面的实时多环系统：

```text
视频/VR/键盘等输入采集环        约 100 Hz
短时运动生成/规划环             约 10 Hz 或更高
策略推理控制环                  50 Hz
低层 Unitree API 关节目标发送    500 Hz
```

关键是所有环路解耦，并使用 latest-data-wins：

- 输入环持续采集最新视频帧，不等待生成模型。
- 生成环只拿最近的观测窗口，不处理已经过期的旧窗口。
- 控制环 50 Hz 稳定运行，永远使用当前最新可用 motion reference。
- 低层 500 Hz 持续发送最新关节目标，避免策略环或输入环短暂抖动阻塞机器人。

论文还提到所有推理和管理组件 onboard 执行，并用 TensorRT 与 CUDA Graph 加速。这里的意思是：模型推理尽量在机器人板载 GPU/CPU 上完成，或者至少在同一实时工作站上完成，避免网络往返和 Python 进程调度导致控制延迟不可控。

## 3. Sliding Window + Overlap Inpainting

GENMO/GEM 是序列生成模型，直接等待完整长视频会产生大延迟。论文采用滑动窗口：

```text
窗口0: frame 0..31
窗口1: frame 24..55    overlap=8
窗口2: frame 48..79    overlap=8
```

窗口之间有 overlap。论文特别说它们修改了 diffusion denoising 过程，在每个 denoising step 中，把新窗口 overlap 部分的 clean motion 覆盖成上一窗口已经生成的 motion。这样做的作用是：

- 新窗口不会在边界处突然跳姿态。
- 每次只需要等待很短的新观测，而不是等待 2 秒、5 秒或完整视频。
- 未观测的未来帧可以用 mask token/empty token，让模型结合 motion prior 生成短 horizon。

注意：这不是简单地在最终 SMPL 结果上拼接 overlap。真正论文级做法是在 diffusion denoising 内部做 inpainting 约束。公开 GENMO demo 默认没有暴露这个实时 scheduler，所以我们新增了 `GENMO/realtime_trt/` 目录来专门攻这个部分。

## 4. Protocol v3 在实时链路里的位置

SONIC deploy 侧仍然接收 ZMQ `pose` topic，消息格式是 Protocol v3。ZMQ 是通信方式，Protocol v3 是 payload schema。

Protocol v3 字段：

```text
joint_pos      [N, 29]
joint_vel      [N, 29]
smpl_joints    [N, 24, 3]
smpl_pose      [N, 21, 3]
body_quat_w    [N, 4]
frame_index    [N]
```

部署侧收到 v3 后进入 SMPL Encode Mode 2。对 GENMO 视频输入来说，主信息来自 SMPL 字段；G1 wrist joints 可以从 SMPL 手臂姿态近似映射，其他 joint_pos/joint_vel 可以先按现有 bridge 的做法生成。

要低延迟，Protocol v3 发布线程应该：

- 50 Hz 独立运行。
- PUB socket 设置低 HWM。
- 非阻塞发送。
- frame_index 单调递增。
- 总是发送最新 motion reference，而不是排队发送旧窗口。

## 5. 当前公开代码和论文内部实现的差距

当前公开 GENMO demo 的路径是：

```text
mp4 -> 读完整窗口 -> YOLO PyTorch/Ultralytics -> ViTPose PyTorch -> HMR2 PyTorch
    -> GEM PyTorch diffusion 50 steps -> smpl_params.pt -> 再转 Protocol v3
```

这个路径适合离线验证，不适合实时控制。它慢的原因包括：

- YOLO/ViTPose/HMR2/GEM 都在 PyTorch eager 路径运行。
- 每个窗口重复做大量 Python 逻辑。
- diffusion 默认 50 step，本身就比单步回归模型重。
- demo 以文件和缓存为中心，不是内存中的实时 pipeline。
- 当前公开代码没有随仓库提供论文使用的 TensorRT engines。
- 当前公开代码没有直接暴露论文式 overlap inpainting scheduler。

所以仅仅把 `TensorRT-10.13.3.9` 放在工作站上，不会自动让 demo 变快。必须把各模型导出成 ONNX/engine，并实现 runtime 调度。

## 6. 新增工程目录

本轮新增两个隔离入口：

```text
$WORKSPACE/GENMO/realtime_trt/
$WORKSPACE/GR00T-WholeBodyControl/local_video_realtime_bridge/realtime_trt_stream.py
```

`GENMO/realtime_trt/` 负责：

- 环境诊断：`diagnose_env.py`
- 安装本地 TensorRT Python wheel：`install_deps.py`
- YOLO 导出：`export_yolo.py`
- ViTPose 导出：`export_vitpose.py`
- HMR2 导出：`export_hmr2.py`
- GEM denoiser 捕获与导出尝试：`export_gem_denoiser.py`
- 一键编排：`build_engines.py`
- runtime 边界和 blocker：`runtime.py`
- profile/readiness 检查：`profile_pipeline.py`

`realtime_trt_stream.py` 负责：

- 接本地 mp4 或相机。
- 检查 TensorRT engine 是否齐全。
- 只走 TensorRT runtime，不回退到 PyTorch demo。
- 缺 engine 或 runtime 未绑定时直接写 blocker 报告。
- 后续 engine 完整后，发布 Protocol v3 到 sim/real。

## 7. 推荐执行顺序

先检查环境：

```bash
cd $WORKSPACE/GENMO
source .venv/bin/activate

python realtime_trt/diagnose_env.py
python realtime_trt/install_deps.py --dry-run
```

如果 dry run 显示 wheel 存在，再安装：

```bash
python realtime_trt/install_deps.py --with-onnxruntime-gpu
```

尝试导出和构建：

```bash
python realtime_trt/build_engines.py \
  --example-video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --build-engines \
  --fp16
```

输出报告默认在：

```text
$WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/
```

如果某一步失败，先看对应 JSON：

```bash
cat $WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/export_gem_denoiser_report.json
cat $WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/build_engines_summary.json
```

然后再启动严格 TensorRT runtime：

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
conda activate sonic

python local_video_realtime_bridge/realtime_trt_stream.py \
  --source video \
  --video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --genmo-repo $WORKSPACE/GENMO \
  --engine-root $WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/engines \
  --engine-window-frames 32 \
  --observed-frames 8 \
  --overlap-frames 8 \
  --target-fps 50 \
  --zmq-host '*' \
  --zmq-port 5556 \
  --zmq-topic pose
```

如果 engine 不齐，它会退出并写：

```text
$WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt/runtime_blocker.json
```

这是预期行为：严格模式不允许偷偷走 PyTorch 慢路径。

## 8. MuJoCo 验证方式

TensorRT engine 和 runtime 完整后，仍然三终端：

Terminal 1:

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
conda activate sonic
python gear_sonic/scripts/run_sim_loop.py
```

Terminal 2:

```bash
cd $WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose sim
```

Terminal 3:

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
conda activate sonic
python local_video_realtime_bridge/realtime_trt_stream.py \
  --source video \
  --video /path/to/input.mp4 \
  --genmo-repo $WORKSPACE/GENMO \
  --engine-root outputs/realtime_trt/engines \
  --zmq-host '*' \
  --zmq-port 5556 \
  --zmq-topic pose
```

Terminal 2 中按：

```text
]      启动控制
ENTER  开启 ZMQ streaming
O      急停/退出
```

## 9. 相机路线接口

真实相机只替换输入源，不改变后端：

```bash
python local_video_realtime_bridge/realtime_trt_stream.py \
  --source camera \
  --camera-index 0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 60 \
  --genmo-repo $WORKSPACE/GENMO \
  --engine-root outputs/realtime_trt/engines \
  --zmq-host '*' \
  --zmq-port 5556 \
  --zmq-topic pose
```

推荐相机：

- 单目 RGB，全局快门优先。
- USB3/UVC 或 GigE/RTSP 均可。
- 720p 60fps 优先；30fps 可用于初步验证。
- 尽量低曝光时间、低压缩延迟。
- 固定机位，保证动作者全身入镜。
- 背景干净、光照稳定、衣服贴身，减少遮挡和关键点抖动。

如果是工业相机/GigE，相机资料最好包括：

- SDK 名称与 Python/C++ API。
- 是否支持 GStreamer。
- 分辨率、帧率、曝光控制方式。
- 输出格式，例如 RGB/BGR/YUYV/H264/RTSP。
- 端到端采集延迟或缓冲机制。

## 10. 验收标准

论文级复现不能只看“程序能跑”。建议按下面指标验收：

- 预热后 `capture_ts -> publish_ts` p95 小于 150 ms。
- 机器人不等待 10 秒级窗口结果。
- 视频发布同步窗口和 MuJoCo 窗口视觉上接近同步。
- ZMQ 发送稳定 50 Hz。
- 旧窗口不会排队，latest-data-wins 生效。
- overlap 边界没有明显姿态跳变。
- 失败时有明确模块级报告，而不是静默降级。

当前新增代码已经把“严格 TensorRT 路线”的门槛建立起来。下一步真正的难点是根据 `export_gem_denoiser_report.json` 的结果，逐个解决 GEM denoiser ONNX 导出、TensorRT binding、DDIM scheduler 和 SMPL postprocess 的接口问题。
