# Ling-3.0-tiny (CUDA)

These recipes export [`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
and [`inclusionAI/Ling-3.0-tiny-fp8`](https://huggingface.co/inclusionAI/Ling-3.0-tiny-fp8)
as ONNX Runtime GenAI packages for the CUDA execution provider using:

- Kimi Delta Attention (KDA) layers via `com.microsoft.VarlenCausalConvWithState`, `com.microsoft.GatedDeltaNet` (`gate_activation="sigmoid"`, `gate_lower_bound=-5.0`, per-key-dim decay), and `com.microsoft.GatedRMSNorm`;
- Multi-Head Latent Attention (MLA) layers via `com.microsoft.PagedAttention` (`head_size=192`, 256-token page blocks) with optional calibrated INT8 per-channel KV cache; and
- Sparse MoE layers (128 routed experts, top-8 `noaux_tc` sigmoid group-limited routing + 1 shared expert) via `com.microsoft.MoE` / `com.microsoft.QMoE`.

## Recipes

1. **BF16 baseline (`inclusionAI-Ling-3.0-tiny_cuda_bf16.json`)**:
   Exports `inclusionAI/Ling-3.0-tiny` in BF16 with paged attention.
2. **Block-FP8 (`inclusionAI-Ling-3.0-tiny_cuda_fp8.json`)**:
   Exports `inclusionAI/Ling-3.0-tiny-fp8` using 128x128 block-scaled `MatMulBlockQuantizedFp8Weight` and `QMoE(quant_type="fp8", block_size=128, swiglu_fusion=2)`.
3. **Quantized INT4 + INT8 KV (`inclusionAI-Ling-3.0-tiny_cuda_q4_int8kv.json`)**:
   Exports `inclusionAI/Ling-3.0-tiny` with INT4 `QMoE` (`block_size=64`), mixed INT4/INT8 symmetric `MatMulNBits` (`last_matmul:int8,mixed_layers:int8,linear_attn:int8`), INT8 embeddings (`GatherBlockQuantized(bits=8)`), and calibrated INT8 per-channel KV cache (`kv_scales_int8_per_channel.json`).
4. **Quantized INT8 + INT8 KV (`inclusionAI-Ling-3.0-tiny_cuda_q8_int8kv.json`)**:
   Exports `inclusionAI/Ling-3.0-tiny` with INT8 `QMoE` (`block_size=64`), INT8 symmetric `MatMulNBits`, INT8 embeddings (`GatherBlockQuantized(bits=8)`), and calibrated INT8 per-channel KV cache (`kv_scales_int8_per_channel.json`).

## Export

Run from this directory so that `kv_scales_int8_per_channel.json` is resolved:

```bash
olive run --config inclusionAI-Ling-3.0-tiny_cuda_q4_int8kv.json
```
