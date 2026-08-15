# Qwen3.6-35B-A3B-NVFP4 (CUDA)

Export [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
to an ONNX Runtime GenAI model. This is the NVFP4 (4-bit) checkpoint of Qwen3.6-35B-A3B,
quantized by [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer):

- **MoE experts + shared expert**: `W4A16_NVFP4`, group size 16 (E2M1 weights, FP8-E4M3
  block scales, per-expert FP32 global scale). Exported directly to the ONNX Runtime
  `com.microsoft.QMoE` op with `quant_type="nvfp4"`, `block_size=16` (no re-quantization).
- Attention projections are FP8 in the source checkpoint.

## Build

```bash
olive run --config qwen3.6-35b-a3b-nvfp4_cuda.json
```

The routed experts are read directly from the source safetensors and packed into the
QMoE NVFP4 layout by the ONNX Runtime GenAI model builder (`--extra_options use_nvfp4_moe=true`).

## Requirements

- An ONNX Runtime CUDA build with FP4 QMoE enabled
  (`onnxruntime_USE_FP4_QMOE=ON`, `ENABLE_FP4`); the `quant_type="nvfp4"` QMoE runs via the
  dequant-fallback path (works on SM90/Hopper such as H200; native block-scaled FP4 GEMM is
  Blackwell-only).
- `onnxruntime-genai` built from a branch that includes the NVFP4 model-builder support
  (`use_nvfp4_moe`).
