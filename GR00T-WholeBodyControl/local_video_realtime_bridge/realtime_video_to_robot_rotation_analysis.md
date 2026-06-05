# 实时视频驱动 G1 机器人转身与稳定性问题分析报告

本文整理当前从视频输入到 G1 机器人执行动作的完整流程，并解释调试过程中出现的几个关键概念：`neutral smpl_pose chunk`、`Reset heading state`、`num_frames_to_send`、`offline` 与 `realtime` 的区别，以及为什么之前的 180 度转身问题可以缓解到 90 度，而当前 90 度转身和不稳定问题不能简单沿用同样方法解决。

本文基于当前调试日志与代码路径整理，重点服务于问题汇报和后续工程定位。

## 1. 总体结论

目前问题可以拆成两个阶段：

1. 之前的 180 度转身问题，主要是起始数据和 heading 初始化问题。
   - 实时链路一开始可能发布了近似 neutral 的 SMPL 数据。
   - 控制端可能用这段不可靠数据初始化 reference root heading。
   - 后续每个 streaming chunk 还可能重复触发 heading reset。
   - 这些会造成 reference 朝向和机器人当前朝向之间出现接近 180 度的错误。

2. 现在剩下的 90 度转身问题，主要不是简单常量偏移，而是 reference root orientation 本身在策略观测中持续驱动机器人转向。
   - 当前日志显示，前几帧 `body_quat` 可能接近初始 heading，但未来帧中 `body_quat` 会快速出现明显 yaw 变化。
   - SMPL 模式下 policy 使用 `smpl_anchor_orientation_*` 作为 root 朝向目标。
   - 因此即使第一帧看起来对齐，policy 仍会根据未来 root orientation 要求机器人转身。

当前不稳定、滑步、朝一侧倾斜，主要来自观测不一致、root yaw 目标过激、短窗口导致未来帧不足，以及 realtime 多窗口预测不连续。

## 2. 从视频到机器人动作的完整流程

当前实时链路大致如下：

```text
输入视频帧
  -> realtime_trt_stream.py 采集视频窗口
  -> TensorRT runtime
       YOLO 检人
       VitPose 估计 2D 关键点
       HMR2 / GEM 条件编码
       GEM denoiser 扩散预测动作
       GEM decode 得到 SMPL 参数
  -> Python bridge 后处理
       smpl_pose
       smpl_joints
       body_quat_w
       joint_pos / joint_vel
  -> ZMQ Protocol v3 发布到 localhost:5556
  -> C++ ZMQEndpointInterface 接收并解码
  -> StreamedMotionMerger 合并 streaming motion
  -> G1Deploy 根据 current_motion 构造 policy observation
  -> ONNX policy 推理
  -> motor command
  -> 机器人执行动作
```

各阶段的关键数据如下。

### 2.1 Python 端输出给 C++ 的数据

`realtime_trt_stream.py` 最终通过 ZMQ 发送 Protocol v3。主要字段包括：

```text
smpl_joints:  [N, 24, 3]
smpl_pose:    [N, 21, 3]
body_quat_w:  [N, 4]
joint_pos:    [N, 29]
joint_vel:    [N, 29]
frame_index:  [N]
```

其中：

- `smpl_pose` 是 SMPL 关节轴角，主要描述人体姿态。
- `smpl_joints` 是 24 个 SMPL joint 的三维位置，目前作为 SMPL 模式下的重要观测。
- `body_quat_w` 是 root/body quaternion，四元数顺序是 `w, x, y, z`。
- `joint_pos` / `joint_vel` 主要是机器人关节目标，目前很多帧中为 0，SMPL 模式主要依赖 SMPL 观测。

### 2.2 C++ 端如何使用这些数据

C++ 端接收后，会将 ZMQ 数据转换为 `MotionSequence`，并把它作为一个名为 `streamed` 的当前 reference motion。

之后 policy 并不是直接读取 ZMQ 原始字段，而是通过 observation gather 函数构造观测。当前 SMPL 模式关键配置在：

```text
gear_sonic_deploy/policy/release/observation_config.yaml
```

SMPL 模式使用的观测包括：

```yaml
required_observations:
  - encoder_mode_4
  - smpl_joints_10frame_step1
  - smpl_anchor_orientation_10frame_step1
  - motion_joint_positions_wrists_10frame_step1
```

当前调试中曾尝试把 orientation 观测替换为：

```yaml
smpl_anchor_orientation_refheading_10frame_step1
```

但目前结果仍然会转身且稳定性不好。

## 3. offline 和 realtime 是什么

这里的 `offline` 和 `realtime` 不是两个机器人，而是同一个视频经过两条不同处理路径后得到的 reference 数据。

### 3.1 offline 数据

offline 数据是提前跑完整离线流程生成的 CSV 文件。例如：

```text
outputs/sonic_motion/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03/
  smpl_joint.csv
  smpl_pose.csv
  body_quat.csv
  joint_pos.csv
  joint_vel.csv
```

它的特点：

- 是提前生成好的完整动作序列。
- 没有实时延迟、窗口切片和 latest-data-wins 问题。
- 可以作为“同一个视频的离线参考版本”。
- 适合用来检查 realtime 输出是否在坐标系、root yaw、SMPL joint 方向上明显偏离。

### 3.2 realtime 数据

realtime 数据是当前实时 TensorRT 链路在线生成的。它的特点：

- 每次只处理一个滑动窗口。
- 每次只发布未来 N 帧，N 由 `num_frames_to_send` 决定。
- 未来动作会随着新窗口不断更新。
- 可能出现窗口之间不连续、启动阶段不可靠、未来帧不足等问题。

### 3.3 为什么比较 offline 第 0 帧和 realtime 第 0 帧

比较 offline 和 realtime 的第一帧，是为了确认：

- 同一个视频在两条链路下，root 朝向是否一致。
- SMPL joints 的局部坐标轴是否一致。
- realtime 是否存在固定的坐标系偏移。

早期诊断中发现：

```text
offline 第 0 帧 body_quat yaw 约 +91.5 deg
realtime 第 0 帧 body_quat yaw 约 -90 deg
二者差值约 178.5 deg
```

这说明当时 realtime 和 offline 之间存在接近 180 度的 root yaw 差异。这个差异不是最终控制结果本身，而是帮助我们判断链路中可能存在坐标系或初始化偏差。

## 4. neutral smpl_pose chunk 是什么

`neutral smpl_pose chunk` 指的是实时系统刚启动时，生成或发布的一段近似“无动作”的 SMPL 数据。

典型表现是：

```text
smpl_pose 接近全 0
body_quat 是固定值或不可靠值
smpl_joints 也可能重复同一帧
```

这类数据不一定代表视频里真实人体动作，而可能来自：

- 启动时窗口还没有足够视频帧。
- diffusion / decode 初始输出不稳定。
- padding 帧较多。
- 第一批 TensorRT 输出还没有进入真实动作状态。

如果这段数据被发布到 C++ 控制端，C++ 端会把它当成真实 reference motion。更严重的是，控制端可能用它执行：

```text
Reset init reference data root rotation to current frame
```

这相当于把一个不可靠的 root quaternion 记录为当前 reference 初始朝向。

因此我们添加了 neutral start gate：

```text
如果 smpl_pose 的 max abs 小于阈值，则暂时不发布
等 smpl_pose 出现真实非零动作后，再打开发布
```

这样做的目的不是修动作本身，而是避免用错误的启动 chunk 初始化 heading。

## 5. Reset heading state 是什么

`Reset heading state` 是 C++ 控制端的一个关键初始化过程。相关逻辑在：

```text
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
UpdateHeadingState()
```

当 `reinitialize_heading_ = true` 时，会做以下事情：

1. 读取机器人当前 IMU/base quaternion。
2. 把当前机器人 heading 存入 `heading_state_buffer_`。
3. 读取当前 reference motion 的 root quaternion。
4. 把它存为 `init_ref_data_root_rot_array_`。

日志形式类似：

```text
Reset heading state to ...
Reset delta heading to 0
Reset init reference data root rotation to current frame: ...
Reference motion name: streamed
```

### 5.1 它为什么重要

控制策略需要知道：

```text
机器人当前朝向
reference 初始朝向
未来 reference 朝向
```

然后计算一个相对 heading，把 reference motion 映射到机器人当前世界方向。

如果 reset 使用了错误 reference 初始帧，或者每个 chunk 都 reset，policy 看到的目标会跳变。机器人会表现为：

- 突然转身。
- 朝某个方向倾斜。
- 脚步滑动。
- 短时间失稳。

### 5.2 之前的问题

之前 streaming update 中，可能因为每个 chunk 的 adjusted frame 或 catch-up 条件导致重复触发 heading reset。

这会导致：

```text
第一个 chunk 用一个 init_ref_root
第二个 chunk 又换一个 init_ref_root
第三个 chunk 又重新对齐
```

对 policy 来说，reference 坐标系在不断变化。

### 5.3 已做的修复

我们将 reset 限制为：

- 刚进入 `streamed` motion 时 reset 一次。
- catch-up reset 时允许 reset。
- 普通后续 chunk 不再因为 frame index 变化重复 reset。

这个改动缓解了“每个 chunk 都重新定义 heading”的问题，是之前从 180 度异常变成较小异常的重要原因之一。

## 6. num_frames_to_send 是什么

`num_frames_to_send` 是 Python 每次通过 ZMQ 发给 C++ 的未来帧数量。

例如：

```bash
--num-frames-to-send 5
```

表示每次消息只发 5 帧。

```bash
--num-frames-to-send 16
```

表示每次消息发 16 帧。

### 6.1 为什么 5 帧不够

SMPL 模式下 policy 需要：

```yaml
smpl_joints_10frame_step1
smpl_anchor_orientation_10frame_step1
```

这表示策略需要看未来 10 帧 SMPL joints 和未来 10 帧 root orientation。

如果每个 ZMQ chunk 只有 5 帧，C++ 端会很快到达当前 streamed motion 末尾，然后出现类似日志：

```text
Motion streamed completed and waiting following motion
```

这表示当前 reference 太短，控制端只能 hold、等待、或者重复使用末尾帧。对 policy 来说，未来动作不完整，稳定性会变差。

### 6.2 为什么改成 16 帧

`num_frames_to_send=16` 至少覆盖了 `10frame_step1` 需要的未来窗口，并留出一定余量。

改成 16 后，日志中每个 chunk 变成：

```text
Decoded smpl_joints: 16 frames, 24 joints
Protocol v3: Received SMPL action (chunk) - frames: 0 to 15
```

这比 5 帧更合理。

但要注意：这只能解决“未来帧不够”的问题，不能解决 reference root yaw 本身要求机器人转身的问题。

## 7. 之前 180 度转身问题的原因分析

早期现象：

```text
机器人开始动作前会先转约 180 度
```

结合日志和对比，可能原因包括以下几类。

### 7.1 offline 和 realtime root yaw 差异接近 180 度

早期对同一视频的 offline CSV 和 realtime 输出做对比：

```text
offline 第 0 帧 body_quat yaw 约 +91.5 deg
realtime 第 0 帧 body_quat yaw 约 -90 deg
差值约 178.5 deg
```

这说明 realtime 生成的 root yaw 和 offline 参考之间存在半圈偏差。

可能来源：

- SMPL base rotation 处理差异。
- y-up / z-up 坐标转换差异。
- root quaternion 的初始化帧不一致。
- 第一批 realtime 输出不是有效动作，而是 neutral/padded chunk。

### 7.2 起始 neutral chunk 参与 heading 初始化

如果第一批发布的数据是 neutral chunk，控制端会把它作为真实动作的起点。此时：

```text
init_ref_data_root_rot_array_
```

可能记录了错误的 root quaternion。

之后真实动作进入时，reference 和机器人当前 heading 的差异会被放大，表现为转身。

### 7.3 每个 streaming chunk 重复 Reset heading state

重复 reset 会让 reference 坐标系不断跳变。即使每次跳变不大，积累起来也会让 policy 一直追逐变化的 root heading。

### 7.4 5 帧 chunk 导致未来观测不足

当 `num_frames_to_send=5` 时，policy 需要的 10 帧未来观测无法完整提供。控制端很快进入 motion completed / waiting 状态，也会让动作执行不平滑。

## 8. 之前 180 度问题的解决过程

已做改动可以总结为四类。

### 8.1 跳过 neutral startup chunk

添加 `NeutralStartGate`：

```text
smpl_pose 太接近 0 -> 不发布
smpl_pose 出现真实动作 -> 开始发布
```

作用：

- 避免错误起始帧初始化 reference heading。
- 让 C++ 第一次收到的 streamed motion 更接近真实动作。

### 8.2 修复重复 heading reset

C++ 端将 heading reset 限制在刚进入 streamed motion 或 catch-up 时触发。

作用：

- 避免每个 chunk 重置 heading。
- 提高 reference 坐标系连续性。

### 8.3 增大 num_frames_to_send

将每次发布帧数从 5 提高到 16。

作用：

- 满足 policy 需要的 10 帧未来观测。
- 减少 motion completed / waiting。

### 8.4 避免直接使用 yaw lock 作为主方案

早期尝试过硬性锁定 root yaw，但它会让 `body_quat`、`smpl_joints`、heading 初始化之间产生新的不一致，容易导致滑步或不稳定。因此不作为主方案。

## 9. 当前 90 度转身问题的原因分析

目前 180 度问题被缓解后，剩下的主要是约 90 度转身和不稳定。

这个问题和之前 180 度问题不同。

### 9.1 当前日志显示未来 root quat 仍在驱动转身

在最新日志中，起始帧可能类似：

```text
body_quat: [(0.998837, -0.009586, -0.038169, 0.027866)]
smpl_joints: [(-0.351225, 0.015799, 0.005778)]
```

这说明第一帧已经不像早期那样有明显 180 度错误。

但后续帧中出现：

```text
body_quat: [(0.953990, ..., 0.297959)]
body_quat: [(0.769789, ..., 0.637511)]
body_quat: [(0.759791, ..., 0.649144)]
```

这些 quaternion 对应明显 root yaw 变化。也就是说，reference motion 的未来帧本身在要求 root 转向。

### 9.2 SMPL 模式包含 root orientation 观测

SMPL 模式不仅看 SMPL joints，还看：

```yaml
smpl_anchor_orientation_10frame_step1
```

或实验中替换过的：

```yaml
smpl_anchor_orientation_refheading_10frame_step1
```

这条观测告诉 policy：未来 root orientation 应该是什么。

如果视频里人物的 root orientation 和机器人当前朝向存在 90 度坐标差，policy 会认为机器人需要转到那个方向。

### 9.3 只转 smpl_joints 不能解决 root orientation

我们尝试过：

```bash
--smpl-joints-yaw-offset-deg -90
```

结果：

- `smpl_joints` 第一帧确实被转到了更接近 offline 的方向。
- 但 `body_quat` 没变，root orientation 观测仍然会要求机器人转身。

因此只修 SMPL 点云坐标，不足以解决 root yaw 驱动。

### 9.4 同时转 body_quat 和 smpl_joints 会破坏策略分布

我们也尝试过：

```bash
--root-frame-yaw-offset-deg 90
--root-frame-yaw-offset-deg -90
```

这会同时改：

```text
body_quat_w
smpl_joints
```

它让部分数值看起来更接近 offline，但控制更不稳定。

原因是 policy 训练时看到的是特定分布的 `smpl_joints + root orientation + robot history` 组合。直接整体旋转可能让某些观测对齐，但让另一些观测和机器人当前状态不一致。

## 10. 为什么不能继续按 180 度方法解决 90 度问题

180 度问题更像是：

```text
起始帧错误 + heading reset 错误 + chunk 太短
```

所以通过：

```text
跳过 neutral chunk
修 reset 时机
增加 chunk 长度
```

可以明显缓解。

现在 90 度问题更像是：

```text
reference 未来 root orientation 本身要求机器人转向
```

这不是启动初始化错误，而是 policy 观测语义问题。

所以继续使用常量 yaw offset 会出现两种失败：

1. 只改 `smpl_joints`：
   - 点云方向变了。
   - root orientation 仍然要求转。

2. 同时改 `body_quat` 和 `smpl_joints`：
   - root orientation 也变了。
   - 但和 heading reset、robot history、policy 训练分布不一致。
   - 容易滑步、倾斜、不稳。

## 11. 不稳定、滑步、朝一侧倾斜的原因分析

当前不稳定不是单一原因，可能由以下因素叠加造成。

### 11.1 观测之间不一致

policy 同时看：

```text
机器人历史状态
SMPL joints
root orientation
手腕 joint target
```

如果我们只改其中一项，会造成语义不一致。

例如：

```text
smpl_joints 被旋转了
body_quat 没旋转
robot current heading 没变
```

policy 会看到一个在局部坐标中改变方向的人体骨架，同时 root orientation 又指向另一个方向。

### 11.2 root yaw 目标过激

视频人物可以在几帧内发生较大朝向变化，但机器人执行需要满足动力学约束。G1 不能瞬间完成大 yaw 转向。

如果 reference 在未来 10 帧内给出快速 yaw 变化，机器人可能通过：

- 扭腰。
- 侧向脚底滑动。
- 身体倾斜。
- 快速转髋。

去追目标，导致不稳定。

### 11.3 realtime 窗口预测不连续

realtime 是滑动窗口预测：

```text
window 0 预测未来 16 帧
window 1 又预测未来 16 帧
window 2 再更新未来 16 帧
```

如果不同窗口预测的 root orientation 不连续，C++ merger 会不断合并新 reference。policy 看到的未来目标会抖动。

### 11.4 chunk 太短导致 motion waiting

之前 `num_frames_to_send=5` 时，当前 motion 很快用完，C++ 日志中出现：

```text
Motion streamed completed and waiting following motion
```

这会造成未来观测不足或 hold 末帧。虽然现在用 16 帧已经缓解，但它是之前不稳定的重要来源。

### 11.5 heading reset 过频

重复 reset 会造成 reference heading 基准变化。这个问题已修复一部分，但仍需要确认当前 C++ binary 确实是最新编译版本。

## 12. 当前建议保留和停止使用的改动

### 12.1 建议保留

建议保留：

```text
NeutralStartGate
只在进入 streamed motion / catch-up 时 reset heading
num_frames_to_send = 16
```

这些改动解决的是链路稳定性基础问题。

### 12.2 建议暂时停止使用

建议不要作为主方案继续使用：

```bash
--lock-initial-yaw
--body-yaw-offset-deg
--root-frame-yaw-offset-deg
```

这些方法容易破坏 `body_quat`、`smpl_joints`、robot state、heading state 之间的一致性。

`--smpl-joints-yaw-offset-deg -90` 可以作为诊断项保留，但目前不能单独解决转身。

## 13. 下一步解决方案建议

下一步建议从“去掉 root yaw 驱动”入手，而不是继续做常量旋转。

### 13.1 方案 A：新增 yaw-invariant SMPL root orientation observation

目标：

```text
让 policy 仍然看到 torso roll/pitch 和姿态信息
但不要强制机器人追视频人物的全局 yaw
```

工程实现方向：

1. 在 C++ 中新增：

```text
smpl_anchor_orientation_noheading_10frame_step1
```

2. 计算时去掉 reference root yaw，只保留相对姿态或 roll/pitch。

3. 在 YAML 中把 SMPL 模式从：

```yaml
smpl_anchor_orientation_10frame_step1
```

替换成：

```yaml
smpl_anchor_orientation_noheading_10frame_step1
```

预期效果：

- 机器人不会被视频人物的绝对朝向强制带着转。
- 仍然可以跟踪身体姿态。
- 比常量 yaw offset 更符合 policy 输入一致性。

风险：

- 如果 policy 训练时强依赖 root yaw，去 yaw 后动作跟踪可能变弱。
- 需要实机验证。

### 13.2 方案 B：root yaw 限速

目标：

```text
允许机器人缓慢转向，但禁止短时间大角速度转身
```

实现方向：

在 Python 发布前或 C++ merger 后，对 `body_quat_w` 的 yaw 做 rate limit，例如：

```text
每秒最多变化 20-30 deg
```

预期效果：

- 减少突然转身。
- 减少滑步和倾斜。

风险：

- root yaw 被限速后，SMPL joints 和 root orientation 可能仍需同步处理。
- 仍需要维护观测一致性。

### 13.3 方案 C：只使用 SMPL joints，不使用 root orientation

目标：

```text
先验证 root orientation 是否是转身主因
```

实现方向：

临时把 SMPL 模式下 root orientation 观测替换成 identity / zero 版本，或者固定为第一帧相对方向。

预期效果：

- 如果转身明显消失，说明 root orientation 是主因。

风险：

- 可能降低上身和躯干朝向跟踪质量。
- 只是诊断方案，不一定是最终方案。

### 13.4 方案 D：构建 offline replay 对照实验

目标：

确认问题来自 realtime 链路还是控制策略本身。

实验：

1. 用同一个视频的 offline CSV 作为 reference motion。
2. 让机器人不经过 realtime TensorRT，直接执行 offline motion。
3. 观察是否仍然出现开始转 90 度。

结论判断：

```text
offline 也转 90 度:
  问题更可能在 SONIC policy / SMPL 模式坐标约定 / reference 本身。

offline 不转，realtime 转:
  问题更可能在 realtime GENMO 输出、窗口拼接、坐标转换、ZMQ 后处理。
```

这个实验很重要，因为它能把问题从“控制端”还是“实时生成端”中分离出来。

## 14. 建议下一步执行顺序

建议按以下顺序推进：

1. 恢复稳定 baseline：
   - 使用 `num_frames_to_send=16`
   - 保留 neutral gate
   - 保留 C++ heading reset 修复
   - 不使用 `root-frame-yaw-offset`、`body-yaw-offset`、`lock-initial-yaw`

2. 做 offline replay 对照：
   - 判断 offline reference 是否也会转 90 度。

3. 增加 C++ debug：
   - 打印 policy 实际使用的 `smpl_anchor_orientation_*` 前几帧。
   - 打印 root yaw 序列。
   - 打印 current robot heading 和 reference heading 差值。

4. 实现 `smpl_anchor_orientation_noheading_10frame_step1`：
   - 去掉绝对 yaw。
   - 保留姿态稳定性信息。

5. 如果仍然不稳，再加入 root yaw rate limiter：
   - 先限制到每秒 20-30 度。
   - 必要时短时间 freeze yaw。

## 15. 汇报摘要

可以用以下口径汇报：

> 当前实时视频到机器人执行链路中，180 度转身问题主要来自启动阶段 neutral SMPL 数据、heading 初始化错误、streaming chunk 太短和重复 heading reset。通过跳过 neutral chunk、修复 heading reset 时机、增加 `num_frames_to_send`，该问题被缓解。现在剩下的 90 度转身不是同类问题，而是 SMPL 模式下 root orientation 观测持续驱动机器人追踪视频人物的全局 yaw。简单常量旋转会破坏 `smpl_joints`、`body_quat`、机器人历史状态和 policy 训练分布之间的一致性，因此会引发滑步、倾斜和不稳定。下一步应通过 offline replay 对照实验定位问题来源，并在 C++ observation 层实现 yaw-invariant 或 yaw-rate-limited 的 root orientation 观测。

