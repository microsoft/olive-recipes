# LiquidAI-LFM2.5-350M — CUDA optimization

## Recipes

### `_cuda_int4.json` — Q4_K_M equivalent
INT4 weights with k_quant_mixed (sensitive layers at INT8) and INT8 embedding/lm_head via RTN.

```
olive run --config LiquidAI-LFM2.5-350M_cuda_int4.json
```

### `_cuda_int8.json` — Q8_0 equivalent
Symmetric INT8 weights with INT8 embedding/lm_head via RTN.

```
olive run --config LiquidAI-LFM2.5-350M_cuda_int8.json
```
