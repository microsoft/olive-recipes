# KEV-4B WebGPU Mobius export

These recipes retain both KEV-4B variants for WebGPU:

```bash
olive run --config kev-4b_webgpu_fp32.json
olive run --config kev-4b_webgpu_mixed.json
```

FP32 publishes to `build-fp32/`; the qualified FP16/INT4 middle-MLP package
publishes to `build-mixed/`. Both emit WebGPU provider metadata and one host
thread per component.

ORT registers WebGPU kernels for Qwen3.5 `LinearAttention`,
`CausalConvWithState`, attention, normalization, and `MatMulNBits`. Runtime
validation still requires a hardware WebGPU/Vulkan adapter. This development
host exposes only Mesa lavapipe, which aborts in LLVM before model execution.
