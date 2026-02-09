# Qwen-Qwen3-14B — CUDA optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-14B targeting the CUDA EP.

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: INT4 + INT8 mixed precision by default
- Example recipe filename: Qwen-Qwen3-14B_cuda_int4_int8_kquant_mixed.json

## Setup

1) Install Olive (version compatible with your repo).
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai-cuda (CUDA build) with compatible CUDA/cuDNN versions
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-14B_cuda_int4_int8_kquant_mixed.json

Additional notes:
- Ensure CUDA and cuDNN versions are compatible with your onnxruntime-genai package.
- Requires an NVIDIA GPU and matching CUDA drivers/toolkit.

---

This README was auto-generated for the CUDA EP of Qwen-Qwen3-14B.
