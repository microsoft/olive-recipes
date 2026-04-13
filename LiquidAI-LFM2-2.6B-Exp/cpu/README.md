# LiquidAI-LFM2-2.6B-Exp — CPU optimization

INT4 quantization via ModelBuilder with accuracy_level=4.
Conv layer projections and their MLPs are automatically promoted to INT8 by the genai builder.

```
olive run --config LiquidAI-LFM2-2.6B-Exp_cpu_int4.json
```
