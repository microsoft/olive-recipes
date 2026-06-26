# gpt-oss-20b — CPU optimization

This folder contains Olive recipes for optimizing gpt-oss-20b targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: INT4 precision by default

## Recipes

- `gpt-oss-20b_cpu_int4_int4_qmoe_default.json`
   - INT4 model with default RTN behavior and `MatMul` + `Gather` quantization.
   - Use this for the standard CPU INT4 build.

- `gpt-oss-20b_cpu_int4_int4_qmoe_k_quant_mixed.json`
   - INT4 model using `int4_algo_config = k_quant_mixed`.
   - Use this when you want the k-quant mixed algorithm variant.

- `gpt-oss-20b_cpu_int4_int8_qmoe_default.json`
   - INT4 model with INT8 QMoE (`use_8bits_moe = true`) and default RTN + `MatMul`/`Gather` quantization.
   - Use this when targeting CPU QMoE with INT8 expert weights.

- `gpt-oss-20b_cpu_int4_int8_qmoe_k_quant_mixed.json`
   - INT4 model with INT8 QMoE (`use_8bits_moe = true`) and `k_quant_mixed`.
   - Use this when you want both INT8 QMoE and the k-quant mixed variant.

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CPU build)
3) Run Olive to build/optimize the model
   - olive run --config gpt-oss-20b_cpu_int4_int4_qmoe_default.json

Additional notes:
- Runs purely on CPU; no GPU required.

---

This README was auto-generated for the CPU EP of gpt-oss-20b.
