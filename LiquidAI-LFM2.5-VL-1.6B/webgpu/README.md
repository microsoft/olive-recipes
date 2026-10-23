# LiquidAI-LFM2.5-VL-1.6B — WebGPU optimization

LFM2.5-VL runs in ONNX Runtime GenAI as three models: the LFM2 decoder taking
`inputs_embeds`, the SigLIP2 vision encoder with its projector, and the embedding
model that splices the image features into the token embeddings. The decoder is
built by the ONNX Runtime GenAI model builder (`exclude_embeds`), exactly like the
LFM2.5 text recipes; the vision encoder and the embedding model are exported by
[Mobius](https://github.com/onnxruntime/mobius). Every recipe writes into `model/`.

## Recipes

### Decoder

#### `_webgpu_int4.json` — INT4 weights
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8.

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_webgpu_int4.json
```

#### `_webgpu_fp16_int4.json` — FP16 LM head + INT4 weights
INT4 weights via RTN, with the LM head excluded so it stays FP16. Larger than
`_webgpu_int4.json`, and less accurate (see below).

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_webgpu_fp16_int4.json
```

### Vision encoder and embedding model

#### `_webgpu_vision_int8.json`
Exports `vision_encoder/` and `embedding/` with Mobius (FP16) and quantizes
both to INT8 (block-wise RTN in blocks of 32; the embedding table becomes
`GatherBlockQuantized`). Shared by both decoder variants. On LFM2.5-VL-450M, the
per-token cosine similarity of the image features to the FP32 export on a test
photo (worst token / mean, CPU EP) is 0.9921 / 0.9997. INT4 is not offered for
the vision encoder: it drops to 0.67 / 0.94. `accuracy_level` is left at 0 for
the same reason: INT8 compute (`accuracy_level: 4`) drops to 0.9233 / 0.9989.

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_webgpu_vision_int8.json
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

Run it with `examples/python/model-mm.py` from onnxruntime-genai. In Python the WebGPU EP ships as
the `onnxruntime-ep-webgpu` plug-in (in `requirements.txt`), which `model-mm.py` registers by itself.
Until [microsoft/onnxruntime-genai#2605](https://github.com/microsoft/onnxruntime-genai/pull/2605)
is released, only text prompts work (see below):

```
python model-mm.py -m model -e WebGpuExecutionProvider
```

## Measured quality

Scored against the Hugging Face model in FP32 on wikitext-2 (test split, 64 chunks of 512 tokens,
second half of each chunk scored): `KLD` is the mean KL divergence of the next-token distribution
from FP32 (lower is better) and `same top` is how often the most likely next token matches FP32. The
llama.cpp rows are LiquidAI's official GGUFs, scored the same way with `llama-perplexity
--kl-divergence`; each range spans its Metal backend and its CPU backend, which quantizes
activations to 8 bits as ONNX Runtime's CPU kernels do. The ONNX rows were measured on an Apple M3
Ultra (Metal). Sizes count the decoder plus the embedding model; the vision encoder is not scored.

| recipe | size | KLD | same top |
| --- | --- | --- | --- |
| `_webgpu_int4.json` | 0.91 GiB | 0.050 | 88.7% |
| `_webgpu_fp16_int4.json` | 0.94 GiB | 0.098 | 84.3% |
| llama.cpp Q4_K_M | 0.68 GiB | 0.036-0.040 | 89.7-90.1% |
| llama.cpp Q8_0 | 1.16 GiB | 0.0003-0.0008 | 98.3-98.9% |

INT4 trails Q4_K_M: 25% more KLD than llama.cpp's CPU backend (0.050 against 0.040), at 1.3x its
size. The size comes from two places: the LM head and the embedding model each hold an INT8 copy of
the token table, where Q4_K_M keeps one 6-bit copy, and MatMulNBits stores an FP16 scale for every
block of 32 weights, where Q4_K packs 6-bit scales into 256-weight super-blocks.

Where INT4 trails, the cause is ONNX Runtime's `k_quant` rather than the recipe: it fits each
block's scale and minimum the way llama.cpp does, then rounds the minimum to an integer zero point
without refitting, which leaves its 4-bit weights with about 1.3x the rounding error of Q4_K. The
recipe's INT8 LM head and INT8 sensitive layers (the same layers Q4_K_M promotes to 6 bits) make up
for part of that. [microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814)
fixes the rounding.

`_webgpu_fp16_int4.json` is the least accurate recipe here despite being the larger one (KLD 0.098
against 0.050): plain RTN costs more than the FP16 LM head saves.

## WebGPU status

Verified with `onnxruntime` 1.30 and the `onnxruntime-ep-webgpu` 0.4.0 plug-in on an Apple M3 Ultra
(Dawn over Metal) and on Linux (Dawn over Vulkan, llvmpipe): the embedding model and the decoder of
every package in this line run on WebGPU, the embedding output matches CPU, and prefill picks the
same next token as the PyTorch model. The 450M package's decoder and embedding model also run in
the browser with `onnxruntime-web` 1.30 on WebGPU (Apple M3 Max).

Three things to know:

- **The vision encoder stays on CPU.** It resizes the SigLIP2 position embeddings with an
  antialiased `Resize`, which the WebGPU EP does not implement ("The antialias attribute of Resize
  operator is NOT implemented") on Metal or Vulkan, plug-in 0.4.0 included. `finalize.py` therefore
  pins that one session to CPU; drop the override once the EP supports the op.
- **Image prompts need [microsoft/onnxruntime-genai#2605](https://github.com/microsoft/onnxruntime-genai/pull/2605),
  which is not merged yet.** Text prompts work without it; see
  [Image prompts in ONNX Runtime GenAI](#image-prompts-in-onnx-runtime-genai) below.
- **ONNX Runtime 1.29 or newer is required.** Older builds do not know the `state_window` attribute
  of `CausalConvWithState` (the LFM2 short-convolution op) and refuse to load the decoder. The
  `onnxruntime` Python package with the plug-in and the `onnxruntime-web` npm package are new
  enough; the standalone `onnxruntime-webgpu` Python wheel is still at 1.27 and is not.

### Image prompts in ONNX Runtime GenAI

A package runs as three sessions: the vision encoder turns the image into features, the embedding
model splices them into the token embeddings, and the decoder generates. `finalize.py` puts the
embedding model and the decoder on WebGPU and keeps the vision encoder on CPU (see above).

GenAI up to and including 0.17.0 allocates the buffer that receives the vision encoder's output on
the decoder's device, so the CPU session is handed a WebGPU buffer as its output. ONNX Runtime
neither copies nor rejects that binding: the CPU kernels write the features over the WebGPU buffer
handle as if it were host memory, and the heap is corrupted. The symptom changes from run to run: a
segfault, a trap, an abort in ONNX Runtime's `BFCArena` allocator, `objc: Method cache corrupted` on
macOS, and now and then a run that completes. Text prompts never run the vision encoder and are
unaffected. The bug is tracked in
[microsoft/onnxruntime-genai#2604](https://github.com/microsoft/onnxruntime-genai/issues/2604)
and is not specific to LFM2-VL: any multimodal model with a sub-model on CPU next to a GPU decoder
hits it.

[microsoft/onnxruntime-genai#2605](https://github.com/microsoft/onnxruntime-genai/pull/2605)
allocates each buffer on the device of the session that writes it and copies it across where the
writing and the reading session differ. With a GenAI build that includes it, every package in this
line describes a test photo correctly on WebGPU (checked on an Apple M3 Ultra).

WebGPU is required rather than merely faster: the wasm backend cannot run the decoder at all
(`GroupQueryAttention` with q/k normalization is implemented only on the CUDA and WebGPU EPs) and
cannot look up the quantized embedding table (no 8-bit `GatherBlockQuantized` kernel).

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
