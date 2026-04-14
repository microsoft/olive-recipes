# LiquidAI-LFM2.5-1.2B-Instruct — CPU optimization

## Recipes

### `_cpu_int4.json` — Q4_K_M equivalent
INT4 weights with k_quant_mixed (sensitive layers at INT8) and INT8 embedding/lm_head via RTN.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_cpu_int4.json
```

### `_cpu_int8.json` — Q8_0 equivalent
Symmetric INT8 weights with INT8 embedding/lm_head via RTN.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_cpu_int8.json
```
