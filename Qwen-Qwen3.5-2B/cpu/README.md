# Qwen-Qwen3.5-2B — CPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3.5-2B targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: INT4 precision by default
- Example recipe filename: Qwen-Qwen3.5-2B_cpu_int4.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CPU build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3.5-2B_cpu_int4.json

Additional notes:
- Pipeline: `ModelBuilder` (INT4 via Neural Compressor) → `QuantizeEmbeddingInt8` (post-hoc INT8 embedding) → `ShareEmbeddingLmHead` (share INT8 weight between embedding and lm_head)
- Model size: ~1.4 GB (down from 4.3 GB FP16)
- Uses text-only mode (exclude_embeds=false) for standalone LLM inference without multimodal pipeline.
- Runs purely on CPU; no GPU required.

---

This README was auto-generated for the CPU EP of Qwen-Qwen3.5-2B.
