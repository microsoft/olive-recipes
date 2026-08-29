# gpt-oss-20b — WebGPU optimization

This folder contains Olive recipes for optimizing gpt-oss-20b targeting the WebGPU EP.

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: INT4 precision by default

## Recipes

- `gpt-oss-20b_webgpu_int4_int4_qmoe_default.json`
   - INT4 model using default RTN behavior with `MatMul` + `Gather` quantization.
   - Use this for the standard WebGPU INT4 build.

- `gpt-oss-20b_webgpu_int4_int4_qmoe_k_quant_mixed.json`
   - INT4 model using `int4_algo_config = k_quant_mixed`.
   - Use this when you want the k-quant mixed variant for WebGPU.

- `gpt-oss-20b_webgpu_int4_int8_qmoe_default.json`
   - INT4 model with `use_8bits_moe = true` and default RTN + `MatMul`/`Gather` quantization.
   - Use this when targeting WebGPU QMoE with INT8 expert weights.

- `gpt-oss-20b_webgpu_int4_int8_qmoe_k_quant_mixed.json`
   - INT4 model with `use_8bits_moe = true` and `k_quant_mixed`.
   - Use this when you want both INT8 QMoE and the k-quant mixed variant on WebGPU.

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-web (WebGPU build)
3) Run Olive to build/optimize the model
   - olive run --config gpt-oss-20b_webgpu_int4_int4_qmoe_default.json

Additional notes:
- WebGPU enables GPU-accelerated inference in web browsers.
- Ensure your browser supports WebGPU (Chrome 113+, Edge 113+).

---

This README was auto-generated for the WebGPU EP of gpt-oss-20b.
