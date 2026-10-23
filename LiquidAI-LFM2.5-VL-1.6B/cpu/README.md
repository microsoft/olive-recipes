# LiquidAI-LFM2.5-VL-1.6B — CPU optimization

LFM2.5-VL runs in ONNX Runtime GenAI as three models: the LFM2 decoder taking
`inputs_embeds`, the SigLIP2 vision encoder with its projector, and the embedding
model that splices the image features into the token embeddings. The decoder is
built by the ONNX Runtime GenAI model builder (`exclude_embeds`), exactly like the
LFM2.5 text recipes; the vision encoder and the embedding model are exported by
[Mobius](https://github.com/onnxruntime/mobius). Every recipe writes into `model/`.

## Recipes

### Decoder

#### `_cpu_int4.json` — INT4 weights
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8.

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_cpu_int4.json
```

#### `_cpu_int8.json` — INT8 weights
Symmetric INT8 weights throughout, including the LM head.

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_cpu_int8.json
```

### Vision encoder and embedding model

#### `_cpu_vision_int8.json`
Exports `vision_encoder/` and `embedding/` with Mobius (FP32) and quantizes
both to INT8 (block-wise RTN in blocks of 128; the embedding table becomes
`GatherBlockQuantized`). Shared by both decoder variants. On LFM2.5-VL-450M, the
per-token cosine similarity of the image features to the FP32 export on a test
photo (worst token / mean) is 0.9953 / 0.9996. INT4 is not offered for the
vision encoder: it drops to 0.53 / 0.92 (0.67 / 0.94 in blocks of 32).
`accuracy_level` is left at 0 for the same reason: INT8 compute
(`accuracy_level: 4`) drops to 0.9439 / 0.9976. Blocks of 32 are no more
accurate at INT8 (0.9895 / 0.9997) and make the files larger.

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_cpu_vision_int8.json
```

### Assemble the package

Adds the `vision` and `embedding` sections to `genai_config.json` and writes
`processor_config.json` (the image preprocessing pipeline, derived from the
Hugging Face `processor_config.json`). Run it after a decoder recipe and the
vision recipe; run it again whenever a decoder recipe rewrites `genai_config.json`.

```
python ../finalize.py
```

The result in `model/`:

```
model/
├── model.onnx (+ .data)            # decoder
├── vision_encoder/model.onnx (+ .data)
├── embedding/model.onnx (+ .data)
├── genai_config.json
├── processor_config.json
└── tokenizer.json, tokenizer_config.json, chat_template.jinja, ...
```

Run it with `examples/python/model-mm.py` from onnxruntime-genai:

```
python model-mm.py -m model -e cpu
```

## Measured quality

Scored against the Hugging Face model in FP32 on wikitext-2 (test split, 64 chunks of 512 tokens,
second half of each chunk scored): `KLD` is the mean KL divergence of the next-token distribution
from FP32 (lower is better) and `same top` is how often the most likely next token matches FP32. The
llama.cpp rows are LiquidAI's official GGUFs, scored the same way with `llama-perplexity
--kl-divergence`; each range spans its Metal backend and its CPU backend, which quantizes
activations to 8 bits as ONNX Runtime's CPU kernels do. The ONNX rows were measured on an Apple M3
Ultra. Sizes count the decoder plus the embedding model; the vision encoder is not scored.

| recipe | size | KLD | same top |
| --- | --- | --- | --- |
| `_cpu_int4.json` | 0.99 GiB | 0.050 | 88.9% |
| `_cpu_int8.json` | 1.39 GiB | 0.0008 | 98.3% |
| llama.cpp Q4_K_M | 0.68 GiB | 0.036-0.040 | 89.7-90.1% |
| llama.cpp Q8_0 | 1.16 GiB | 0.0003-0.0008 | 98.3-98.9% |

INT8 matches Q8_0 (0.0008 against 0.0008).

INT4 trails Q4_K_M: 25% more KLD than llama.cpp's CPU backend (0.050 against 0.040), at 1.5x its
size. The size comes from two places: the LM head and the embedding model each hold an INT8 copy of
the token table, where Q4_K_M keeps one 6-bit copy, and MatMulNBits stores an FP32 scale for every
block of 32 weights, where Q4_K packs 6-bit scales into 256-weight super-blocks.

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

These recipes need Olive from `main`, Mobius from git (`requirements.txt` pins the
commit they were verified with: the released `mobius-onnx` has neither LFM2-VL nor
the `revision` argument Olive passes), and `onnxruntime-genai>=0.17.0`, the first
release with LFM2-VL support
([microsoft/onnxruntime-genai#2571](https://github.com/microsoft/onnxruntime-genai/pull/2571))
in both the model builder and the runtime. The released `olive-ai` package cannot
drive genai 0.15+ (its ModelBuilder pass skips the `check_extra_options` step that
`create_model` now requires).

Image splitting (tiling) is not implemented in ONNX Runtime GenAI: every image is
resized once to at most `max_image_tokens` tokens, as the Hugging Face processor
does with `do_image_splitting=false`.
