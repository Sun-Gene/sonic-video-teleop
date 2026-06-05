from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TensorBinding:
    name: str
    mode: str
    dtype: str
    shape: tuple[int, ...]


class TensorRTEngineRunner:
    """Small TensorRT v10 runner using cuda-python bindings.

    The class intentionally keeps host buffers visible and returns numpy arrays.
    That makes it useful for runtime debugging and for comparing TensorRT output
    against PyTorch before we move more of GENMO into device-only execution.
    """

    def __init__(self, engine_path: str | Path, *, profile_index: int = 0) -> None:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.trt = trt
        self.cudart = cudart
        self.engine_path = Path(engine_path).expanduser().resolve()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with self.engine_path.open("rb") as handle, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context: {self.engine_path}")

        self.profile_index = int(profile_index)
        err, stream = cudart.cudaStreamCreate()
        self._check_cuda(err)
        self.stream = stream
        self.bindings: list[TensorBinding] = []
        self._buffers: dict[str, tuple[np.ndarray, int, int]] = {}
        self._allocate_static_buffers()

    def close(self) -> None:
        for host, device, _nbytes in self._buffers.values():
            self._check_cuda(self.cudart.cudaFree(device)[0])
            self._check_cuda(self.cudart.cudaFreeHost(host.ctypes.data)[0])
        self._buffers.clear()
        if getattr(self, "stream", None) is not None:
            self._check_cuda(self.cudart.cudaStreamDestroy(self.stream)[0])
            self.stream = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": binding.name,
                "mode": binding.mode,
                "dtype": binding.dtype,
                "shape": list(binding.shape),
            }
            for binding in self.bindings
        ]

    @property
    def input_names(self) -> list[str]:
        return [b.name for b in self.bindings if b.mode == "INPUT"]

    @property
    def output_names(self) -> list[str]:
        return [b.name for b in self.bindings if b.mode == "OUTPUT"]

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        kind_h2d = self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        kind_d2h = self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost

        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT input '{name}' for {self.engine_path.name}")
            host, device, nbytes = self._buffers[name]
            array = np.asarray(inputs[name], dtype=host.dtype)
            if array.size != host.size:
                raise ValueError(
                    f"Input '{name}' expected {host.size} values shaped "
                    f"{self._binding_shape(name)}, got {array.shape}"
                )
            np.copyto(host.reshape(array.shape), array, casting="safe")
            err = self.cudart.cudaMemcpyAsync(device, host, nbytes, kind_h2d, self.stream)[0]
            self._check_cuda(err)

        for binding in self.bindings:
            _host, device, _nbytes = self._buffers[binding.name]
            if not self.context.set_tensor_address(binding.name, int(device)):
                raise RuntimeError(f"Failed to bind tensor address: {binding.name}")

        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")

        outputs: dict[str, np.ndarray] = {}
        for name in self.output_names:
            host, device, nbytes = self._buffers[name]
            err = self.cudart.cudaMemcpyAsync(host, device, nbytes, kind_d2h, self.stream)[0]
            self._check_cuda(err)
            outputs[name] = host.copy().reshape(self._binding_shape(name))

        self._check_cuda(self.cudart.cudaStreamSynchronize(self.stream)[0])
        return outputs

    def _allocate_static_buffers(self) -> None:
        trt = self.trt
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(int(v) for v in self.engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                profile_shape = self.engine.get_tensor_profile_shape(name, self.profile_index)[-1]
                shape = tuple(int(v) for v in profile_shape)
                self.context.set_input_shape(name, shape)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            size = int(trt.volume(shape))
            nbytes = int(size * dtype.itemsize)
            err, host_ptr = self.cudart.cudaMallocHost(nbytes)
            self._check_cuda(err)
            err, device_ptr = self.cudart.cudaMalloc(nbytes)
            self._check_cuda(err)
            import ctypes

            c_type = np.ctypeslib.as_ctypes_type(dtype)
            host = np.ctypeslib.as_array(ctypes.cast(host_ptr, ctypes.POINTER(c_type)), (size,))
            self._buffers[name] = (host, int(device_ptr), nbytes)
            self.bindings.append(
                TensorBinding(
                    name=name,
                    mode="INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT",
                    dtype=str(dtype),
                    shape=shape,
                )
            )

    def _binding_shape(self, name: str) -> tuple[int, ...]:
        for binding in self.bindings:
            if binding.name == name:
                return binding.shape
        raise KeyError(name)

    @staticmethod
    def _check_cuda(err: Any) -> None:
        from cuda.bindings import runtime as cudart

        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA runtime error: {err}")
