# Qwen-Qwen3-4B-Instruct-2507 — WEBGPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-4B-Instruct-2507 targeting the WebGPU EP.

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: INT4 + INT8 mixed precision by default
- Example recipe filename: Qwen-Qwen3-4B-Instruct-2507_webgpu_int4_int8_kquant_mixed.json

## Setup

1) Install Olive (version compatible with your repo).
2) Install the appropriate runtime package for this backend:
   - onnxruntime-webgpu and onnxruntime-genai
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-4B-Instruct-2507_webgpu_int4_int8_kquant_mixed.json

Additional notes:
- Ensure onnxruntime-genai is installed with the --no-deps flag. Otherwise, it will install the CPU build of ONNX Runtime and override your WebGPU build.
- Runs in a WebGPU-capable environment.

---

This README was auto-generated for the WEBGPU EP of Qwen-Qwen3-4B-Instruct-2507.
