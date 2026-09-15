# LiquidAI-LFM2.5-350M — WebGPU optimization

## Recipes

### `_webgpu_int4.json` — Q4_K_M equivalent
INT4 weights via k_quant_mixed: sensitive layers and the LM head are kept at INT8,
and the embedding table stays FP16.

```
olive run --config LiquidAI-LFM2.5-350M_webgpu_int4.json
```

### `_webgpu_fp16_int4.json` — FP16 embedding + INT4 weights
INT4 weights via RTN, with the LM head excluded so both it and the embedding
table stay FP16. Larger than `_webgpu_int4.json`, but highest accuracy.

```
olive run --config LiquidAI-LFM2.5-350M_webgpu_fp16_int4.json
```
