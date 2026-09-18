# LiquidAI-LFM2.5-VL-3B — CUDA optimization

LFM2.5-VL runs in ONNX Runtime GenAI as three models: the LFM2 decoder taking
`inputs_embeds`, the SigLIP2 vision encoder with its projector, and the embedding
model that splices the image features into the token embeddings. The decoder is
built by the ONNX Runtime GenAI model builder (`exclude_embeds`), exactly like the
LFM2.5 text recipes; the vision encoder and the embedding model are exported by
[Mobius](https://github.com/onnxruntime/mobius). Every recipe writes into `model/`.

## Recipes

### Decoder

#### `_cuda_int4.json` — Q4_K_M equivalent
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8.

```
olive run --config LiquidAI-LFM2.5-VL-3B_cuda_int4.json
```

#### `_cuda_int8.json` — Q8_0 equivalent
Symmetric INT8 weights throughout, including the LM head.

```
olive run --config LiquidAI-LFM2.5-VL-3B_cuda_int8.json
```

### Vision encoder and embedding model

#### `_cuda_vision_int8.json`
Exports `vision_encoder/` and `embedding/` with Mobius (FP16) and quantizes
both to INT8 (block-wise RTN; the embedding table becomes `GatherBlockQuantized`).
Shared by both decoder variants. INT4 is not offered for the vision encoder: on
LFM2.5-VL-450M it drops the per-token cosine similarity of the image features to
~0.94, while INT8 keeps it above 0.99. `accuracy_level` is left at 0 for the same
reason — INT8 compute (`accuracy_level: 4`) roughly doubles the feature error
(per-token cosine 0.9902 vs 0.9971 at worst on LFM2.5-VL-450M).

```
olive run --config LiquidAI-LFM2.5-VL-3B_cuda_vision_int8.json
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
python model-mm.py -m model -e cuda
```

LFM2.5-VL-3B ships a `tokenizer.json` whose pre-tokenizer pattern
(`'(?i:[sdmt]|ll|ve|re)|...`) the tokenizer in onnxruntime-extensions cannot
parse. `finalize.py` replaces it with the equivalent pattern the other LFM2.5
models use (verified to produce identical tokens on chat prompts, code and 20k
random strings), so `og.Tokenizer(model)` loads the package as is.

## Setup

Python 3.11+ is required: onnxruntime-genai stopped publishing cp310 wheels at 0.12.

```
pip install git+https://github.com/microsoft/olive.git
pip install -r requirements.txt
```

These recipes need Olive from `main`, Mobius from `main` (`requirements.txt` points
at git: the released `mobius-onnx` has neither LFM2-VL nor the `revision` argument
Olive passes), and
an `onnxruntime-genai-cuda` build that includes LFM2-VL support
([microsoft/onnxruntime-genai#2571](https://github.com/microsoft/onnxruntime-genai/pull/2571)),
both in the model builder and in the runtime. The released `olive-ai` package
cannot drive genai 0.15+ (its ModelBuilder pass skips the `check_extra_options`
step that `create_model` now requires), and Olive `main` imports the
`onnxruntime_genai.models.loaders` package that only ships from genai 0.16.0.

Image splitting (tiling) is not implemented in ONNX Runtime GenAI: every image is
resized once to at most `max_image_tokens` tokens, as the Hugging Face processor
does with `do_image_splitting=false`.
