# Nemotron Scripts

Shared utilities and tests for the Nemotron 3.5 ASR Streaming Multilingual
0.6B recipes.

ONNX export is handled through the provider-specific Olive recipes:

- [`../cpu/`](../cpu) for CPU
- [`../NvTensorRtRtx/`](../NvTensorRtRtx) for TRT-RTX

From the recipe directory, run one of:

```bash
python cpu/optimize.py
python NvTensorRtRtx/optimize.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `export_tokenizer.py` | Extract the NeMo vocabulary and create an ORT-compatible tokenizer |
| `test_optimize.py` | Validate provider-specific export selection and model-loader behavior |
