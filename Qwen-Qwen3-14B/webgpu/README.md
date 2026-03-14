# Qwen-Qwen3-14B — WebGPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-14B targeting the WebGPU EP.

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: INT4 precision by default
- Example recipe filename: Qwen-Qwen3-14B_webgpu_int4.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-web (WebGPU build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-14B_webgpu_int4.json

Additional notes:
- This model uses `k_quant_mixed` via the `SelectiveMixedPrecision` pass followed by
  `GPTQ` and `ModelBuilder`, instead of the `kld_gradient` algorithm used by smaller
  Qwen3 models (0.6B–8B). The `kld_gradient` algorithm requires loading the full model
  to GPU for gradient-based sensitivity estimation, which exceeds the 80 GB per-GPU
  memory limit for the 14B model. The `k_quant_mixed` algorithm uses a pre-defined
  quantization sensitivity map and does not require GPU memory for sensitivity estimation.
- GPTQ group size: 32
- WebGPU enables GPU-accelerated inference in web browsers.
- Ensure your browser supports WebGPU (Chrome 113+, Edge 113+).

---

This README was auto-generated for the WebGPU EP of Qwen-Qwen3-14B.
