# LiquidAI-LFM2.5-1.2B-Thinking — WebGPU optimization

## Recipes

### `_webgpu_int4.json` — Q4_K_M equivalent
INT4 weights via k_quant_mixed: sensitive layers and the LM head are kept at INT8,
and the embedding table stays FP16.

```
olive run --config LiquidAI-LFM2.5-1.2B-Thinking_webgpu_int4.json
```

### `_webgpu_fp16_int4.json` — FP16 embedding + INT4 weights
INT4 weights via RTN, with the LM head excluded so both it and the embedding
table stay FP16. Larger than `_webgpu_int4.json`, but highest accuracy.

```
olive run --config LiquidAI-LFM2.5-1.2B-Thinking_webgpu_fp16_int4.json
```

## Requirements

Needs Python 3.11+ and `onnxruntime-genai>=0.15.0`. LFM2 support landed in
genai 0.14.0, and 0.15.0 renamed the quantization option this recipe uses
(`int4_algo_config` to `algo_config`). genai stopped publishing cp310 wheels
at 0.12, so on Python 3.10 pip silently resolves to 0.11.4 and the build
fails with `The LiquidAI/... model is not currently supported`.
