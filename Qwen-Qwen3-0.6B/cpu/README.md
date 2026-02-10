# Qwen-Qwen3-0.6B — CPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-0.6B targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: INT4 precision by default
- Example recipe filename: Qwen-Qwen3-0.6B_cpu_int4_kld_gradient.json

## Setup

1) Install Olive (version compatible with your repo).
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CPU build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-0.6B_cpu_int4_kld_gradient.json

Additional notes:
- Optional: Use best practices when considering accuracy vs. memory to improve throughput on CPU.
- Runs purely on CPU; no GPU required.

---

This README was auto-generated for the CPU EP of Qwen-Qwen3-0.6B.
