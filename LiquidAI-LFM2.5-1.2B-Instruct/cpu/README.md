# LiquidAI-LFM2.5-1.2B-Instruct — CPU optimization

INT4 quantization via ModelBuilder with accuracy_level=4 (INT8 activations for MatMul).
Conv layer projections and their MLPs are automatically promoted to INT8 by the genai builder.

```
olive run --config LiquidAI-LFM2.5-1.2B-Instruct_cpu_int4.json
```
