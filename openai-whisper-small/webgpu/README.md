# openai-whisper-small — WebGPU optimization

This folder contains Olive recipes for optimizing openai-whisper-small targeting the WebGPU EP.

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: INT8 precision by default
- Example recipe filename: whisper-small_webgpu_int8.json

## Setup

1) Install Olive:
   - pip install olive-ai
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (WebGPU build)
3) Run Olive to build/optimize the model
   - olive run --config whisper-small_webgpu_int8.json

Additional notes:
- Pipeline: `ModelBuilder` (fp16) → `OnnxDynamicQuantization` (int8, MatMul/Gemm/Gather)
- WebGPU enables GPU-accelerated inference in web browsers.
- Ensure your browser supports WebGPU (Chrome 113+, Edge 113+).

---

This README was auto-generated for the WebGPU EP of openai-whisper-small.
