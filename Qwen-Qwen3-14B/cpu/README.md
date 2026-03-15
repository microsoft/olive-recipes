# Qwen-Qwen3-14B — CPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-14B targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: INT4 precision by default
- Example recipe filename: Qwen-Qwen3-14B_cpu_int4.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CPU build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-14B_cpu_int4.json

Additional notes:
- Pipeline: `SelectiveMixedPrecision` (k_quant_mixed) → `GPTQ` → `RTN` (8-bit lm_head/embeddings) → `ModelBuilder`
- Uses `k_quant_mixed` instead of `kld_gradient` because gradient-based sensitivity
  estimation exceeds the 80 GB per-GPU memory limit for the 14B model.
- GPTQ group size: 128
- Runs purely on CPU; no GPU required.

---

This README was auto-generated for the CPU EP of Qwen-Qwen3-14B.
