# GENMO TensorRT Realtime 工程线

这个目录是隔离的 TensorRT 复现工程入口，不替换 GENMO 官方 demo。它的目标是把论文里“视频观测 -> GENMO/GEM -> 低延迟 motion”的路线拆成可验证的导出、构建、诊断和 runtime 组件。

当前阶段的原则：

- 成功导出的模块会生成 ONNX/TensorRT engine 和 JSON 报告。
- 导出失败会写明 blocker，不会把 PyTorch demo 慢路径伪装成论文级实时。
- 真正 `<150ms` 的验收必须依赖完整 TensorRT engine 加 runtime 调度。

推荐先运行：

```bash
cd $WORKSPACE/GENMO
source .venv/bin/activate

python realtime_trt/diagnose_env.py
python realtime_trt/install_deps.py --dry-run
```

如果 dry run 显示 TensorRT wheel 路径正确，再手动安装。脚本会先尝试 `python -m pip`；如果这个 uv venv 里没有 pip，会自动改用 `uv pip install --python <当前解释器>`：

```bash
python realtime_trt/install_deps.py
```

`onnxruntime-gpu` 只用于后续数值对比，可按需安装：

```bash
python realtime_trt/install_deps.py --with-onnxruntime-gpu
```

构建尝试：

```bash
python realtime_trt/build_engines.py \
  --example-video $WORKSPACE/dataset/mp4/excerpt/lite/gBR_sBM_c01_d04_mBR3_ch03.mp4 \
  --build-engines \
  --fp16
```

输出默认写到：

```text
$WORKSPACE/GR00T-WholeBodyControl/outputs/realtime_trt
```
