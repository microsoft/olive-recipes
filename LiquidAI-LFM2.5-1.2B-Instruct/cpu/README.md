# LiquidAI-LFM2.5-1.2B-Instruct — CPU optimization

## Recipes

### `_cpu_int4.json` — Q4_K_M equivalent
INT4 weights via k_quant_mixed: sensitive layers and the LM head are kept at INT8,
and the embedding table is left unquantized.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_cpu_int4.json
```

### `_cpu_int8.json` — Q8_0 equivalent
Symmetric INT8 weights throughout, with the embedding table and LM head also at INT8 via RTN.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_cpu_int8.json
```
