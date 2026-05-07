# openai-whisper-large-v2 — CUDA optimization

This folder contains Olive recipes for optimizing openai-whisper-large-v2 targeting the CUDA EP.

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: INT8 precision by default
- Example recipe filename: whisper-large-v2_cuda_int8.json

## Setup

1) Install Olive:
   - pip install olive-ai
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CUDA build)
3) Run Olive to build/optimize the model
   - olive run --config whisper-large-v2_cuda_int8.json

Additional notes:
- Sets all MatMul nodes to 8-bit using k-quant.
- Pipeline: `ModelBuilder` (fp16) → `OnnxKQuantQuantization` (k-quant INT8 MatMul weights)
- Requires NVIDIA GPU with CUDA support.
- Ensure CUDA toolkit and cuDNN are properly installed.

---

This README was auto-generated for the CUDA EP of openai-whisper-large-v2.
