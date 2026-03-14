# Qwen-Qwen3-4B — CUDA optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-4B targeting the CUDA EP.

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: INT4 precision by default
- Example recipe filename: Qwen-Qwen3-4B_cuda_int4.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai-cuda (CUDA build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-4B_cuda_int4.json

Additional notes:
- Requires NVIDIA GPU with CUDA support.
- Ensure CUDA toolkit and cuDNN are properly installed.

---

This README was auto-generated for the CUDA EP of Qwen-Qwen3-4B.
