# openai-whisper-small — CUDA optimization

This folder contains Olive recipes for optimizing openai-whisper-small targeting the CUDA EP.

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: INT8 precision by default
- Example recipe filename: whisper-small_cuda_int8.json

## Setup

1) Install Olive:
   - pip install olive-ai
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CUDA build)
3) Run Olive to build/optimize the model
   - olive run --config whisper-small_cuda_int8.json

Additional notes:
- Pipeline: `ModelBuilder` (fp16) → `OnnxDynamicQuantization` (int8, MatMul/Gemm/Gather)
- Requires NVIDIA GPU with CUDA support.
- Ensure CUDA toolkit and cuDNN are properly installed.

---

This README was auto-generated for the CUDA EP of openai-whisper-small.
