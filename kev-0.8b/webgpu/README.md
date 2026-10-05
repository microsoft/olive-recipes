# KEV-0.8B WebGPU Mobius export

All retained KEV-0.8B package variants have WebGPU recipes:

```bash
olive run --config kev-0.8b_webgpu_fp32.json
olive run --config kev-0.8b_webgpu_fp16.json
olive run --config kev-0.8b_webgpu_mixed.json
```

They publish independently to `build-fp32/`, `build-fp16/`, and
`build-mixed/`. FP16 is the selected CUDA precision; mixed INT4 is retained as
an experimental variant because its calibration regressed in full-suite
qualification.

ORT registers WebGPU kernels for the complete Qwen3.5 component graph. Runtime
validation requires a hardware WebGPU/Vulkan adapter. This development host
exposes only Mesa lavapipe, which aborts in LLVM before model execution.
