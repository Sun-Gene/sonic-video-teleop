# SONIC Video Teleoperation

Realtime video-to-motion teleoperation overlay for SONIC whole-body control.

This repository contains the extra code needed to stream local video motion
through GENMO TensorRT inference and publish SONIC ZMQ Protocol v3 references
for MuJoCo simulation or real G1 deployment.

## Language

- [中文说明](README.zh-CN.md)
- [English Guide](README.en.md)

## Demo

Demo media is stored in `assets/demo/`, which is a common convention for images
and GIFs shown by README files. The `examples/` directory is usually better for
runnable examples or sample scripts.

<p align="center">
  <img src="assets/demo/demo1.gif" width="48%" alt="Video teleoperation demo 1" />
  <img src="assets/demo/demo2.gif" width="48%" alt="Video teleoperation demo 2" />
</p>

## Upstream Projects

- GR00T-WholeBodyControl: https://github.com/NVlabs/GR00T-WholeBodyControl
- GENMO: https://github.com/NVlabs/GENMO

Large generated artifacts are intentionally not committed: TensorRT engines,
ONNX files, checkpoints, videos, and runtime outputs should be rebuilt locally.
