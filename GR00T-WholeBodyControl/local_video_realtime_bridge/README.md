# 本地视频/真实相机到 SONIC 实时跟踪实验桥

这个目录只放实验代码，尽量不改 GENMO 和 SONIC 官方源码。目标链路是：

```text
mp4 / 单目相机
  -> 滑动窗口
  -> 常驻 GENMO/GEM worker
  -> smpl_params.pt
  -> SONIC Protocol v3
  -> ZMQ pose topic
  -> MuJoCo sim / real
```

重要说明：论文里的实时视频遥操作使用 TensorRT、CUDA Graph、滑动窗口 overlap/inpainting 和板载多频率控制环。这里先复现开源可实现的近似版本：GENMO 主模型常驻，只加载一次 `gem_smpl.ckpt`；窗口之间做 overlap 并丢弃重复前缀；GENMO 慢于实时的时候采用 latest-data-wins，不阻塞视频窗口。

## 文件说明

```text
realtime_stream.py      主入口：本地视频/相机 -> GENMO worker -> ZMQ Protocol v3
genmo_worker.py         用 GENMO .venv 启动的常驻 worker，只加载一次 GEM 主模型
realtime_trt_stream.py  严格 TensorRT 路线入口：不回退 PyTorch，engine 缺失就写 blocker
tensorrt_try_export.py  TensorRT/ONNX 尝试导出和诊断脚本，实验性
video_segment_stream.py 旧版分段脚本，保留作为对照
sonic_video_teleop_tensorrt_design.md 论文级 TensorRT 视频遥操作设计说明
```

## 论文级 TensorRT 路线

如果目标是论文级实时，不要继续使用 `realtime_stream.py` 做验收。它是 PyTorch 近似路线，会受到 GENMO demo 推理速度限制。

严格路线请先阅读：

```bash
less $WORKSPACE/GR00T-WholeBodyControl/local_video_realtime_bridge/sonic_video_teleop_tensorrt_design.md
```

然后在 GENMO 环境中尝试导出和构建 engine：

```bash
cd $WORKSPACE/GENMO
source .venv/bin/activate

python realtime_trt/diagnose_env.py
python realtime_trt/install_deps.py --dry-run
python realtime_trt/build_engines.py \
  --example-video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --build-engines \
  --fp16
```

构建完成后再运行严格入口：

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
conda activate sonic

python local_video_realtime_bridge/realtime_trt_stream.py \
  --source video \
  --video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --genmo-repo $WORKSPACE/GENMO \
  --engine-root outputs/realtime_trt/engines \
  --zmq-host '*' \
  --zmq-port 5556 \
  --zmq-topic pose
```

这个入口不会偷偷使用 PyTorch 慢路径。如果 TensorRT engine 或 GEM runtime binding 不完整，它会退出并写：

```text
outputs/realtime_trt/runtime_blocker.json
```

## 第 1 步：本地 mp4 -> MuJoCo sim

开三个终端。

Terminal 1，启动 MuJoCo：

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
conda activate sonic
python gear_sonic/scripts/run_sim_loop.py
```

Terminal 2，启动 SONIC deploy：

```bash
cd $WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose sim
```

Terminal 3，启动本地视频实时桥：

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
conda activate sonic

python local_video_realtime_bridge/realtime_stream.py \
  --source video \
  --video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --genmo-repo $WORKSPACE/GENMO \
  --genmo-python $WORKSPACE/GENMO/.venv/bin/python \
  --ckpt-path $WORKSPACE/GENMO/inputs/pretrained/gem_smpl.ckpt \
  --hmr2-ckpt $WORKSPACE/GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt \
  --window-seconds 2.0 \
  --overlap-seconds 0.5 \
  --target-fps 50 \
  --zmq-host '*' \
  --zmq-port 5556 \
  --zmq-topic pose \
  --preview-mode publish \
  --preview-scale 0.5
```

Terminal 2 中按键：

```text
]      启动控制
ENTER  开启 ZMQ streaming
O      急停/退出
```

如果只是检查数据格式、不想真的发给 deploy：

```bash
python local_video_realtime_bridge/realtime_stream.py \
  --source video \
  --video /path/to/input.mp4 \
  --max-windows 1 \
  --dry-run-zmq \
  --no-preview
```

## 第 2 步：real 复用方式

第 3 步暂不新增专门代码。第 1/2 步的发布端已经是 ZMQ Protocol v3，部署端从 `sim` 换成 `real` 即可：

```bash
cd $WORKSPACE/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh --input-type zmq --zmq-host <publisher-machine-ip> --zmq-port 5556 --zmq-topic pose real
```

发布端仍然运行 `realtime_stream.py`。实机前必须先用同一视频/相机在 sim 完整验证，现场保留 `O` 急停。

## 关键参数

```text
--window-seconds       每次送给 GENMO 的滑动窗口长度
--overlap-seconds      相邻窗口重叠长度；发布时后续窗口会丢弃 overlap 前缀
--target-fps           SONIC 发布频率，默认 50Hz
--num-frames-to-send   每条 Protocol v3 消息携带的未来帧窗口，默认 5
--process-width/height 可选，降低送入 GENMO 的视频分辨率以减少延迟
--preview-mode         publish 默认：视频窗口与 ZMQ 发布同步；capture 显示原始实时输入流
--max-windows          调试用，只处理前 N 个窗口
--dry-run-zmq          构建 payload 但不发送，打印 Protocol v3 shape
--write-overlay        让 worker 额外生成 2D keypoint overlay；会增加耗时
```

Protocol v3 字段：

```text
joint_pos      [N, 29]
joint_vel      [N, 29]
smpl_joints    [N, 24, 3]
smpl_pose      [N, 21, 3]
body_quat_w    [N, 4]
frame_index    [N]
```

v3 会让部署侧进入 SMPL Encode Mode 2。

## TensorRT 尝试

先做诊断：

```bash
cd $WORKSPACE/GR00T-WholeBodyControl
$WORKSPACE/GENMO/.venv/bin/python local_video_realtime_bridge/tensorrt_try_export.py
```

尝试 ONNX smoke export：

```bash
$WORKSPACE/GENMO/.venv/bin/python local_video_realtime_bridge/tensorrt_try_export.py \
  --try-smoke-export \
  --example-video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4
```

如果已经有 ONNX 文件，尝试构建 engine：

```bash
$WORKSPACE/GENMO/.venv/bin/python local_video_realtime_bridge/tensorrt_try_export.py \
  --build-engine \
  --onnx-out /path/to/model.onnx \
  --engine-out /path/to/model.engine \
  --fp16
```

注意：这个脚本是实验性诊断。GENMO 公开 demo 的 `model.predict()` 包含动态采样和字典输入，可能不能直接导出成可部署 TensorRT engine。失败时脚本会写出 `export_failure.txt`。

## 延迟解释

这个版本比旧的 `video_segment_stream.py` 少了一个最大开销：不会每个窗口重新启动 Python 并重新加载 `gem_smpl.ckpt`。但 ViTPose/HMR2/GENMO 推理仍然可能慢于实时。

默认 `--preview-mode publish` 会在 ZMQ 发布动作时播放对应的视频窗口，方便肉眼比较 MuJoCo 和源视频是否同步。如果改成 `--preview-mode capture`，视频窗口会显示原始输入流；当 GENMO 慢于实时输入时，原始视频会先播完，机器人随后才补播迟到的动作，这种显示方式不适合判断机器人和已发布 motion 的同步性。

如果日志中出现：

```text
drop queued window ...
switch window A -> B
lag=...
```

说明 GENMO 跟不上输入源，系统会丢弃旧窗口，尽量让机器人跟踪最新可用的人体 motion。这是论文里 latest-data-wins 思路的开源近似。

## 安全注意

- 先 `--dry-run-zmq --max-windows 1` 检查环境。
- 再跑 MuJoCo sim。
- sim 稳定后才考虑 real。
- real 时必须有安全员盯住 deploy 终端，手放在 `O` 急停附近。
- 不要第一次就跑大幅跳跃、快速旋转、下肢激烈舞蹈。
