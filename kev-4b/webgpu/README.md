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
`CausalConvWithState`, attention, normalization, and `MatMulNBits`.

## A100 setup and results

Use the NVIDIA EGL Vulkan ICD described in the CLM WebGPU README and install
`onnxruntime-ep-webgpu==0.4.0`.

Thirty in-process requests on one A100 80 GB produced:

| Variant | p50 | p95 | Requests/s | GPU memory delta |
|---|---:|---:|---:|---:|
| FP32 | **228.36 ms** | **267.36 ms** | **4.34** | 16,330 MiB |
| Mixed | 265.56 ms | 298.91 ms | 3.75 | **5,807 MiB** |

The mixed package is slower through WebGPU but uses 64% less GPU memory.
Against the existing CUDA/Foundry service result (39.85 ms p50), WebGPU mixed
is 6.7x slower; llama.cpp BF16 and Q4_K_M service results are 44.69 ms and
46.44 ms respectively.

Full WebGPU qualification evaluated 3,568 records and 4,389 questions with no
rejections or truncation. Relative to the qualified CUDA mixed package, accuracy
changed by -0.37 to +0.46 percentage points; maximum NLL delta was +0.00052 and
maximum Brier delta was +0.000063. The selected answers remained aligned.
