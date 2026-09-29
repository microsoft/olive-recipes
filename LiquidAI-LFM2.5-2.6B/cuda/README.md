# LiquidAI-LFM2.5-2.6B — CUDA optimization

## Recipes

### `_cuda_int4.json` — INT4 weights
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table is left unquantized.

```
olive run --config LiquidAI-LFM2.5-2.6B_cuda_int4.json
```

### `_cuda_int8.json` — INT8 weights
Symmetric INT8 weights throughout, with the embedding table and LM head also at INT8 via RTN.

```
olive run --config LiquidAI-LFM2.5-2.6B_cuda_int8.json
```

## Measured quality

Scored against the Hugging Face model in FP32 on wikitext-2 (test split, 64 chunks of 512 tokens,
second half of each chunk scored): `KLD` is the mean KL divergence of the next-token distribution
from FP32 (lower is better) and `same top` is how often the most likely next token matches FP32. The
llama.cpp rows are LiquidAI's official GGUFs, scored the same way with `llama-perplexity
--kl-divergence`; each range spans its Metal backend and its CPU backend, which quantizes
activations to 8 bits as ONNX Runtime's CPU kernels do. The ONNX rows were measured on an NVIDIA
A10. Sizes count the decoder.

| recipe | size | KLD | same top |
| --- | --- | --- | --- |
| `_cuda_int4.json` | 2.25 GiB | 0.165 | 81.6% |
| `_cuda_int8.json` | 2.94 GiB | 0.0051 | 96.5% |
| llama.cpp Q4_K_M | 1.56 GiB | 0.154-0.155 | 81.3-81.4% |
| llama.cpp Q8_0 | 2.68 GiB | 0.0012-0.0029 | 97.5-98.3% |

INT8 is close to Q8_0 but not level with it (0.0051 against 0.0029 on llama.cpp's CPU backend); the
CUDA EP computes in FP16.

INT4 trails Q4_K_M: 6% more KLD than llama.cpp's CPU backend (0.165 against 0.155), at 1.4x its
size. The size comes from two places: the embedding table stays in FP16 next to the INT8 LM head,
where Q4_K_M keeps one 6-bit copy, and MatMulNBits stores an FP16 scale for every block of 32
weights, where Q4_K packs 6-bit scales into 256-weight super-blocks.

Where INT4 trails, the cause is ONNX Runtime's `k_quant` rather than the recipe: it fits each
block's scale and minimum the way llama.cpp does, then rounds the minimum to an integer zero point
without refitting, which leaves its 4-bit weights with about 1.3x the rounding error of Q4_K. The
recipe's INT8 LM head and INT8 sensitive layers (the same layers Q4_K_M promotes to 6 bits) make up
for part of that. [microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814)
fixes the rounding.

## Setup

Python 3.11+ is required: onnxruntime-genai stopped publishing cp310 wheels at 0.12.

```
pip install git+https://github.com/microsoft/olive.git
pip install -r requirements.txt
```

These recipes need Olive from `main` together with `onnxruntime-genai-cuda>=0.16.0`.
The released `olive-ai` package cannot drive genai 0.15+ (its ModelBuilder pass
skips the `check_extra_options` step that `create_model` now requires), and Olive
`main` imports the `onnxruntime_genai.models.loaders` package that only ships from
genai 0.16.0. LFM2 support itself landed in genai 0.14.0.

The `onnxruntime-genai-cuda` 0.17.0 and `onnxruntime-gpu` 1.30 wheels on PyPI are CUDA 13
builds, so they need NVIDIA driver 580 or newer.
