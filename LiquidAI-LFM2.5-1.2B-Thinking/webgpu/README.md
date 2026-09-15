# LiquidAI-LFM2.5-1.2B-Thinking — WebGPU optimization

## Recipes

### `_webgpu_int4.json` — Q4_K_M equivalent
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table stays FP16.

```
olive run --config LiquidAI-LFM2.5-1.2B-Thinking_webgpu_int4.json
```

### `_webgpu_fp16_int4.json` — FP16 embedding + INT4 weights
INT4 weights via RTN, with the LM head excluded so both it and the embedding
table stay FP16. Larger than `_webgpu_int4.json`, but highest accuracy.

```
olive run --config LiquidAI-LFM2.5-1.2B-Thinking_webgpu_fp16_int4.json
```

## Setup

Python 3.11+ is required: onnxruntime-genai stopped publishing cp310 wheels at 0.12.

```
pip install git+https://github.com/microsoft/olive.git
pip install -r requirements.txt
```

These recipes need Olive from `main` together with `onnxruntime-genai>=0.16.0`.
The released `olive-ai` package cannot drive genai 0.15+ (its ModelBuilder pass
skips the `check_extra_options` step that `create_model` now requires), and Olive
`main` imports the `onnxruntime_genai.models.loaders` package that only ships from
genai 0.16.0. LFM2 support itself landed in genai 0.14.0.
