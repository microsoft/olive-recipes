# tencent-HY-MT1.5-1.8B — CPU optimization

This folder contains Olive recipes for optimizing tencent-HY-MT1.5-1.8B targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: INT4 precision by default
- Example recipe filename: tencent-HY-MT1.5-1.8B_cpu_int4.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CPU build)
3) Run Olive to build/optimize the model
   - olive run --config tencent-HY-MT1.5-1.8B_cpu_int4.json

Additional notes:
- Pipeline: `SelectiveMixedPrecision` (kld_gradient) → `GPTQ` → `RTN` (8-bit lm_head/embeddings) → `ModelBuilder` → `TieWordEmbeddings`
- GPTQ group size: 128
- Runs purely on CPU; no GPU required.

---

This README was auto-generated for the CPU EP of tencent-HY-MT1.5-1.8B.
