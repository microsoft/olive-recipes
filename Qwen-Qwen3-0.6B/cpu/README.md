# Qwen-Qwen3-0.6B — CPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-0.6B targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: FP32 precision by default
- Example recipe filename: Qwen-Qwen3-0.6B_cpu_fp32.json

## Setup

1) Install Olive (version compatible with your repo).
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CPU build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-0.6B_cpu_fp32.json

Additional notes:
- Optional: Use INT4 or INT4 + INT8 quantization recipes to improve throughput on CPU.
- Runs purely on CPU; no GPU required.

---

This README was auto-generated for the CPU EP of Qwen-Qwen3-0.6B.
