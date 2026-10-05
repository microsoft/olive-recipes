# LiquidAI-LFM2.5-350M — CPU optimization

## Recipes

### `_cpu_int4.json` — INT4 weights
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table is left unquantized.

```
olive run --config LiquidAI-LFM2.5-350M_cpu_int4.json
```

### `_cpu_int8.json` — INT8 weights
Symmetric INT8 weights throughout, with the embedding table and LM head also at INT8 via RTN.

```
olive run --config LiquidAI-LFM2.5-350M_cpu_int8.json
```

## Measured quality

KL divergence of the next-token distribution from the FP32 model on wikitext-2 (`KLD`, lower is
better) and how often the most likely next token matches FP32 (`same top`), next to LiquidAI's GGUFs
on llama.cpp's CPU and Metal backends. ONNX rows: the CPU EP on an Apple M3 Ultra. `±` is the
standard error over the 64 scored chunks; sizes count the decoder. Method, scripts and the
measurements behind the notes: [eval](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md).

| recipe | size | KLD | same top |
| --- | --- | --- | --- |
| `_cpu_int4.json` | 0.54 GiB | 0.492 ± 0.012 | 67.5% |
| `_cpu_int8.json` | 0.47 GiB | 0.00896 ± 0.00021 | 95.4% |
| llama.cpp Q4_K_M, CPU | 0.21 GiB | 0.481 ± 0.010 | 67.4% |
| llama.cpp Q4_K_M, Metal | 0.21 GiB | 0.465 ± 0.010 | 68.0% |
| llama.cpp Q8_0, CPU | 0.35 GiB | 0.00843 ± 0.00018 | 95.3% |
| llama.cpp Q8_0, Metal | 0.35 GiB | 0.00370 ± 0.00009 | 96.7% |

- INT4 has 2% ± 1% more KLD than Q4_K_M on llama.cpp CPU and 6% ± 2% more than on Metal (paired on
  the same tokens), at 2.6x its size
  ([why](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md#size)).
- Built with [microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814)
  (open), which fits `k_quant`'s scale to the zero point it stores, INT4 scores 0.468: 3% ± 2% less
  KLD than Q4_K_M on llama.cpp CPU
  ([details](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md#int4-against-q4_k_m-k_quants-zero-point)).
- INT8 has 6% ± 1% more KLD than Q8_0 on llama.cpp CPU and 2.4x Metal's.

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
