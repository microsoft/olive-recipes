# LiquidAI-LFM2.5-1.2B-Instruct — CUDA optimization

## Recipes

### `_cuda_int4.json` — Direct INT4 via ModelBuilder
Simple INT4 quantization. Conv layer projections are automatically promoted to INT8 by the genai builder.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_cuda_int4.json
```

### `_cuda_int4_mixed.json` — Mixed precision with RTN
Pipeline: `SelectiveMixedPrecision` (k_quant_down) → `GPTQ` (INT4, group_size=32) → `RTN` (INT8 embeddings/lm_head) → `ModelBuilder`.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_cuda_int4_mixed.json
```
