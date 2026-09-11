# LiquidAI-LFM2.5-1.2B-Instruct — WebGPU optimization

## Recipes

### `_webgpu_int4.json` — Fully INT4
All weights and embedding quantized to INT4 via RTN. Smallest model size.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_webgpu_int4.json
```

### `_webgpu_fp16_int4.json` — FP16 embedding + INT4 weights
FP16 embedding for better accuracy, INT4 weights via ModelBuilder.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_webgpu_fp16_int4.json
```
