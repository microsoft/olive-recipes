# LiquidAI-LFM2.5-VL-1.6B — WebGPU optimization

LFM2.5-VL runs in ONNX Runtime GenAI as three models: the LFM2 decoder taking
`inputs_embeds`, the SigLIP2 vision encoder with its projector, and the embedding
model that splices the image features into the token embeddings. The decoder is
built by the ONNX Runtime GenAI model builder (`exclude_embeds`), exactly like the
LFM2.5 text recipes; the vision encoder and the embedding model are exported by
[Mobius](https://github.com/onnxruntime/mobius). Every recipe writes into `model/`.

## Recipes

### Decoder

#### `_webgpu_int4.json` — Q4_K_M equivalent
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8.

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_webgpu_int4.json
```

#### `_webgpu_fp16_int4.json` — FP16 LM head + INT4 weights
INT4 weights via RTN, with the LM head excluded so it stays FP16. Larger than
`_webgpu_int4.json`, but highest accuracy.

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_webgpu_fp16_int4.json
```

### Vision encoder and embedding model

#### `_webgpu_vision_int8.json`
Exports `vision_encoder/` and `embedding/` with Mobius (FP16) and quantizes
both to INT8 (block-wise RTN; the embedding table becomes `GatherBlockQuantized`).
Shared by both decoder variants. INT4 is not offered for the vision encoder: on
LFM2.5-VL-450M it drops the per-token cosine similarity of the image features to
~0.94, while INT8 keeps it above 0.99.

```
olive run --config LiquidAI-LFM2.5-VL-1.6B_webgpu_vision_int8.json
```

### Assemble the package

Adds the `vision` and `embedding` sections to `genai_config.json` and writes
`processor_config.json` (the image preprocessing pipeline, derived from the
Hugging Face `processor_config.json`). Run it after a decoder recipe and the
vision recipe; run it again whenever a decoder recipe rewrites `genai_config.json`.

```
python finalize.py
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
python model-mm.py -m model -e webgpu
```

## WebGPU status

The vision encoder runs on CPU even in a WebGPU package: the SigLIP2 position embeddings are
resized with an antialiased `Resize`, which the WebGPU EP does not implement ("The antialias
attribute of Resize operator is NOT implemented"). `finalize.py` therefore leaves that one session
on CPU; the decoder and the embedding model stay on WebGPU. Drop the override in `finalize.py`
once the EP supports the op.

The decoder also needs an `onnxruntime` with WebGPU that is new enough to know the `state_window`
attribute of `CausalConvWithState` (the LFM2 short-convolution op). The newest published WebGPU
build at the time of writing, `onnxruntime-webgpu` 1.27, predates it and refuses to load the
decoder; the same graph loads fine on CPU and CUDA with ORT 1.30. The embedding model was verified
to run on the WebGPU EP.

## Setup

Python 3.11+ is required: onnxruntime-genai stopped publishing cp310 wheels at 0.12.

```
pip install git+https://github.com/microsoft/olive.git
pip install git+https://github.com/onnxruntime/mobius.git
pip install -r requirements.txt
```

These recipes need Olive from `main`, Mobius from `main` (the released
`mobius-onnx` has neither LFM2-VL nor the `revision` argument Olive passes), and
an `onnxruntime-genai` build that includes LFM2-VL support
([microsoft/onnxruntime-genai#2571](https://github.com/microsoft/onnxruntime-genai/pull/2571)),
both in the model builder and in the runtime. The released `olive-ai` package
cannot drive genai 0.15+ (its ModelBuilder pass skips the `check_extra_options`
step that `create_model` now requires), and Olive `main` imports the
`onnxruntime_genai.models.loaders` package that only ships from genai 0.16.0.

Image splitting (tiling) is not implemented in ONNX Runtime GenAI: every image is
resized once to at most `max_image_tokens` tokens, as the Hugging Face processor
does with `do_image_splitting=false`.
