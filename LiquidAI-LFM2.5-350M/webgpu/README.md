# LiquidAI-LFM2.5-350M — WebGPU optimization

## Recipes

### `_webgpu_int4.json` — INT4 weights
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table stays FP16.

```
olive run --config LiquidAI-LFM2.5-350M_webgpu_int4.json
```

### `_webgpu_fp16_int4.json` — FP16 embedding + INT4 weights
INT4 weights via RTN, with the LM head excluded so both it and the embedding
table stay FP16. Larger than `_webgpu_int4.json`, and less accurate (see below).

```
olive run --config LiquidAI-LFM2.5-350M_webgpu_fp16_int4.json
```

## Measured quality

Scored against the Hugging Face model in FP32 on wikitext-2 (test split, 64 chunks of 512 tokens,
second half of each chunk scored): `KLD` is the mean KL divergence of the next-token distribution
from FP32 (lower is better) and `same top` is how often the most likely next token matches FP32. The
llama.cpp rows are LiquidAI's official GGUFs, scored the same way with `llama-perplexity
--kl-divergence`; each range spans its Metal backend and its CPU backend, which quantizes
activations to 8 bits as ONNX Runtime's CPU kernels do. The ONNX rows were measured on an Apple M3
Ultra (Metal). Sizes count the decoder.

| recipe | size | KLD | same top |
| --- | --- | --- | --- |
| `_webgpu_int4.json` | 0.38 GiB | 0.491 | 67.3% |
| `_webgpu_fp16_int4.json` | 0.42 GiB | 0.979 | 55.2% |
| llama.cpp Q4_K_M | 0.21 GiB | 0.465-0.481 | 67.4-68.0% |
| llama.cpp Q8_0 | 0.35 GiB | 0.0037-0.0084 | 95.3-96.7% |

INT4 matches Q4_K_M's quality (0.491 against 0.481 on llama.cpp's CPU backend), at 1.8x its size.
The size comes from two places: the embedding table stays in FP16 next to the INT8 LM head, where
Q4_K_M keeps one 6-bit copy, and MatMulNBits stores an FP16 scale for every block of 32 weights,
where Q4_K packs 6-bit scales into 256-weight super-blocks.

Where INT4 trails, the cause is ONNX Runtime's `k_quant` rather than the recipe: it fits each
block's scale and minimum the way llama.cpp does, then rounds the minimum to an integer zero point
without refitting, which leaves its 4-bit weights with about 1.3x the rounding error of Q4_K. The
recipe's INT8 LM head and INT8 sensitive layers (the same layers Q4_K_M promotes to 6 bits) make up
for part of that. [microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814)
fixes the rounding.

`_webgpu_fp16_int4.json` is the least accurate recipe here despite being the larger one (KLD 0.979
against 0.491): plain RTN costs more than the FP16 LM head saves.

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
