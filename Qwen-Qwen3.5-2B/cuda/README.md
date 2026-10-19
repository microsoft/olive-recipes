# Qwen-Qwen3.5-2B — CUDA optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3.5-2B targeting the CUDA EP.

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: INT4 precision by default
- Example recipe filename: Qwen-Qwen3.5-2B_cuda_int4.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai-cuda (CUDA build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3.5-2B_cuda_int4.json

Additional notes:
- Pipeline: `ModelBuilder` (INT4 via Neural Compressor) → `QuantizeEmbeddingInt8` (post-hoc INT8 embedding) → `ShareEmbeddingLmHead` (share INT8 weight between embedding and lm_head)
- Model size: ~1.4 GB (down from 4.3 GB FP16)
- MMLU accuracy: 57.11% (vs 59.27% FP16 baseline)
- Uses text-only mode (exclude_embeds=false) for standalone LLM inference without multimodal pipeline.
- CUDA graph enabled for optimized decode throughput.
- Requires NVIDIA GPU with CUDA support.
- Ensure CUDA toolkit and cuDNN are properly installed.

---

This README was auto-generated for the CUDA EP of Qwen-Qwen3.5-2B.
