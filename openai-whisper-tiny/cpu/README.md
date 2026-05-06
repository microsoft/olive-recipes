# openai-whisper-tiny — CPU optimization

This folder contains Olive recipes for optimizing openai-whisper-tiny targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: INT8 precision by default
- Example recipe filename: whisper-tiny_cpu_int8.json

## Setup

1) Install Olive:
   - pip install olive-ai
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai
3) Run Olive to build/optimize the model
   - olive run --config whisper-tiny_cpu_int8.json

Additional notes:
- Pipeline: `ModelBuilder` (fp32) → `OnnxDynamicQuantization` (int8, MatMul/Gemm/Gather)
- Runs purely on CPU; no GPU required.

---

This README was auto-generated for the CPU EP of openai-whisper-tiny.
