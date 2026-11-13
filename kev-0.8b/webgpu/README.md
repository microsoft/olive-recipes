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

ORT registers WebGPU kernels for the complete Qwen3.5 component graph.

## A100 setup and results

Use the NVIDIA EGL Vulkan ICD described in the CLM WebGPU README and install
`onnxruntime-ep-webgpu==0.4.0`.

Thirty in-process requests on one A100 80 GB produced:

| Variant | p50 | p95 | Requests/s | GPU memory delta |
|---|---:|---:|---:|---:|
| FP32 | 115.86 ms | 142.01 ms | **8.42** | 3,051 MiB |
| FP16 | **108.25 ms** | 165.83 ms | 8.30 | 1,558 MiB |
| Mixed experimental | 117.09 ms | **150.80 ms** | 8.32 | **1,318 MiB** |

FP16 remains the selected WebGPU variant because it has the lowest p50 and
passed precision qualification. The mixed variant retains its calibration
caveat.

Full WebGPU FP16 qualification evaluated 3,568 records and 4,389 questions with
no rejections or truncation. Relative to CUDA FP16, accuracy changed by -0.09
to +0.40 percentage points; WebGPU NLL and Brier improved on all four suites.

The same model reached 7.41 ms p50 through CUDA FP16 and 188.89 ms on CPU FP32.
No official llama.cpp GGUF or vLLM reference implementation is available for
KEV-0.8B, so those comparisons are unavailable rather than inferred.
