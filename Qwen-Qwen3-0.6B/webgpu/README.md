# Qwen-Qwen3-0.6B — WebGPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-0.6B targeting the WebGPU EP.

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: INT4 precision by default
- Example recipe filename: Qwen-Qwen3-0.6B_webgpu_int4.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-web (WebGPU build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-0.6B_webgpu_int4.json

Additional notes:
- WebGPU enables GPU-accelerated inference in web browsers.
- Ensure your browser supports WebGPU (Chrome 113+, Edge 113+).

---

This README was auto-generated for the WebGPU EP of Qwen-Qwen3-0.6B.
