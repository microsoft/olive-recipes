# CLM-v0.1-8B WebGPU Mobius export

These recipes retain both CLM package variants for WebGPU:

```bash
olive run --config CLM-v0.1-8B_webgpu_fp32.json
olive run --config CLM-v0.1-8B_webgpu_mixed.json
```

FP32 publishes to `build-fp32/`; the FP16/INT4 middle-MLP package publishes to
`build-mixed/`. Download the pinned CLM artifact into `artifacts/clm/` as
described by the CPU/CUDA recipes.

The exported graphs use only operators registered by the ORT WebGPU provider,
including weight-only `MatMulNBits` for the mixed package. Runtime validation
requires a hardware WebGPU/Vulkan adapter. This development host exposes only
Mesa lavapipe; Dawn aborts in LLVM before model execution, so real WebGPU
latency and parity must be measured on supported hardware.
