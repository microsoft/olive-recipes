# gpt-oss-20b — WebGPU optimization

This folder contains Olive recipes for optimizing gpt-oss-20b targeting the WebGPU EP.

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: INT4 precision by default

## Recipes

- `gpt-oss-20b_webgpu_int4_int4_qmoe_default.json`
   - INT4 dense and expert weights using default RTN behavior with `MatMul` + `Gather` quantization.
   - Use this for the standard WebGPU INT4 build.

- `gpt-oss-20b_webgpu_int4_int4_qmoe_k_quant_mixed.json`
   - INT4 dense and expert weights using k-quant with INT8 mixed layers and LM head.
   - Use this when you want the k-quant mixed variant for WebGPU.

- `gpt-oss-20b_webgpu_int4_int4_qmoe_k_quant_mixed_paged.json`
    - The same k-quant mixed weights with WebGPU PagedAttention, 256-token prefill chunks, and a
   reduced block ring for the 12 local-attention layers. Its 1024 global blocks provide
   262,144 aggregate token slots without relying on unavailable WebGPU memory telemetry.
    - Requires the GenAI `Engine` API and a WebGPU plugin with local-window and head-sink
       PagedAttention support.

- `gpt-oss-20b_webgpu_int4_int8_qmoe_default.json`
   - INT4 dense weights with INT8 expert weights and default RTN `MatMul`/`Gather` quantization.
   - Use this when targeting WebGPU QMoE with INT8 expert weights.

- `gpt-oss-20b_webgpu_int4_int8_qmoe_k_quant_mixed.json`
   - INT4 dense weights with INT8 experts using k-quant and INT8 mixed layers and LM head.
   - Use this when you want both INT8 QMoE and the k-quant mixed variant on WebGPU.

All variants keep the MoE router MatMuls in FP16 and emit WebGPU's raw blockwise weight layouts:

- `matmulnbits_weights_prepacked=0`; CUDA fpA-intB MatMul prepacking is disabled.
- `qmoe_weights_prepacked=-1`; the CUDA-only QMoE prepack attribute is omitted.
- KV cache tensors remain FP16. Builder-level INT8 KV cache is not supported by WebGPU.
- The local attention window is preserved, but the KV allocation is full-length in non-paged recipes.
   WebGPU does not support `sliding_window_cache=1`; the paged recipe instead uses a repeating block
   table for local layers.
- WebGPU graph capture is not enabled by default. It changes the generated attention-mask graph and should be
  validated separately for the intended prompt and generation shapes.

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install ONNX Runtime 1.24.4 or newer and the WebGPU plugin EP:
   - `pip install "onnxruntime>=1.24.4"`
   - `pip install -r requirements.txt -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/`
3) Run Olive to build/optimize the model
   - `olive run --config gpt-oss-20b_webgpu_int4_int4_qmoe_default.json`

Additional notes:
- The plugin package runs WebGPU natively through ONNX Runtime. On Linux it requires `libvulkan.so.1` and a
  compatible Vulkan adapter.
- Register the plugin library before creating an ONNX Runtime or ONNX Runtime GenAI session. The GenAI
  `benchmark_e2e.py --execution_provider webgpu` harness performs this registration automatically.

## Exported candidates

All five recipes exported successfully from `openai/gpt-oss-20b`.

| Candidate | QMoE | Dense MatMulNBits | External data bytes | External data SHA-256 |
|---|---:|---:|---:|---|
| `model_int4_int4_qmoe_default` | 24 x INT4 block 32 | 49 x INT4 block 32 | 11,798,016,000 | `f2b32763f50f02fdb112d0adfd199237dfd7c5f9428294cb14b5dd34880a12c6` |
| `model_int4_int4_qmoe_k_quant_mixed` | 24 x INT4 block 32 | 36 x INT4 + 13 x INT8 block 32 | 13,038,960,640 | `86280c006992dca61a0fa5c1d4d3e5d5c86282df9946b193e3d7e29883207a65` |
| `model_int4_int4_qmoe_k_quant_mixed_paged` | 24 x INT4 block 32 | 36 x INT4 + 13 x INT8 block 32 | 13,038,960,640 | `86280c006992dca61a0fa5c1d4d3e5d5c86282df9946b193e3d7e29883207a65` |
| `model_int4_int8_qmoe_default` | 24 x INT8 block 32 | 49 x INT4 block 32 | 21,353,201,664 | `4338c83962086b609fecb20e9a72f71f8ae2287f70b375a8ea0f23bbe0c3ab2a` |
| `model_int4_int8_qmoe_k_quant_mixed` | 24 x INT8 block 32 | 36 x INT4 + 13 x INT8 block 32 | 22,594,109,440 | `06a870b88381d9dbe22bb93c6d43e09819575eae5b2a707abe3c8f6a51f54a8e` |

Every dense graph has 276 nodes, including 24 `GroupQueryAttention` nodes. The 12 local-attention
layers have `local_window_size=128`; the other 12 use `-1`.

The paged graph has 268 nodes and replaces those attention nodes with 24 `PagedAttention` nodes.
It uses `block_table` on the 12 global layers, `block_table_windowed` on the 12 local layers, FP16
paged K/V buffers, learned head-sink initializers, and no `attention_metadata` graph input. A
two-request GenAI Engine smoke test completed through the native Vulkan plugin.

The consolidated graph audit verified:

- No `MatMulNBits.weight_prepacked` attribute.
- No `QMoE.weights_prepacked` attribute.
- No MXFP4 or NVFP4 QMoE encoding; WebGPU uses integer INT4 or INT8 experts.
- Block size 32 for every quantized dense and expert weight.
- FP16 past/present K/V tensors (48 graph inputs); no builder-level quantized KV cache.
- No `sliding_window_cache` attribute. Local attention is windowed, but KV allocation remains full length.
- WebGPU graph capture is off (`enableGraphCapture=0`, `validationMode=basic`).

Block 32 is supported by the WebGPU implementation and is its optimized common layout:
`MatMulNBitsQkv` requires block 32, the fused MLP decode path specializes on block 32, and
`QMoE` delegates expert GEMMs to the same `ApplyMatMulNBits` implementation. Model Builder rejects
MXFP4/NVFP4 QMoE on non-CUDA execution providers.
