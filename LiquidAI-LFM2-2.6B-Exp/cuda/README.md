# LiquidAI-LFM2-2.6B-Exp — CUDA optimization

## Recipes

### `_cuda_int4.json` — Q4_K_M equivalent
INT4 weights via k_quant_mixed: sensitive layers and the LM head are kept at INT8,
and the embedding table is left unquantized.

```
olive run --config LiquidAI-LFM2-2.6B-Exp_cuda_int4.json
```

### `_cuda_int8.json` — Q8_0 equivalent
Symmetric INT8 weights throughout, with the embedding table and LM head also at INT8 via RTN.

```
olive run --config LiquidAI-LFM2-2.6B-Exp_cuda_int8.json
```

## Requirements

Needs Python 3.11+ and `onnxruntime-genai>=0.15.0`. LFM2 support landed in
genai 0.14.0, and 0.15.0 renamed the quantization option this recipe uses
(`int4_algo_config` to `algo_config`). genai stopped publishing cp310 wheels
at 0.12, so on Python 3.10 pip silently resolves to 0.11.4 and the build
fails with `The LiquidAI/... model is not currently supported`.
