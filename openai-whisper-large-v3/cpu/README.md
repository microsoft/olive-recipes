# openai-whisper-large-v3 — CPU optimization

This folder contains Olive recipes for optimizing openai-whisper-large-v3 targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: INT8 precision by default
- Example recipe filename: whisper-large-v3_cpu_int8.json

## Setup

1) Install Olive:
   - pip install olive-ai
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai
3) Run Olive to build/optimize the model
   - olive run --config whisper-large-v3_cpu_int8.json

Additional notes:
- Sets all MatMul nodes to 8-bit using k-quant.
- Pipeline: `ModelBuilder` (fp32) → `OnnxKQuantQuantization` (k-quant INT8 MatMul weights)
- Runs purely on CPU; no GPU required.

---

This README was auto-generated for the CPU EP of openai-whisper-large-v3.
