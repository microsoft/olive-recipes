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
including weight-only `MatMulNBits` for the mixed package.

## A100 setup

The A100 driver exposes EGL but does not install an NVIDIA Vulkan ICD manifest.
Create one without Docker:

```bash
cat >/tmp/nvidia_icd.json <<'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0",
    "api_version": "1.3.0"
  }
}
EOF
export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
```

Validated with `onnxruntime-ep-webgpu==0.4.0` on one A100 80 GB. Thirty
in-process requests produced:

| Variant | Cache-miss p50 | Fully cached p50 | Requests/s, miss | GPU memory delta |
|---|---:|---:|---:|---:|
| FP32 | 233.00 ms | 9.14 ms | 4.25 | 29,029 MiB |
| Mixed | **186.67 ms** | **8.41 ms** | **5.36** | **8,616 MiB** |

The fully cached path still executes CLM scoring. For comparison, the existing
CUDA/Foundry mixed result is 22.87 ms p50 for a changing state with cached
actions, while vLLM BF16 is 29.58 ms. These are different runtime/service
boundaries; WebGPU is reported as an in-process provider result.
