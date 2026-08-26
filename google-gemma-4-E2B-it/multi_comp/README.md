# Gemma 4 E2B — Quantize Then Export

This recipe quantizes the Torch decoder component of
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
before exporting the complete multimodal model with Mobius.

The flow has two explicit stages:

1. Olive selects the Gemma 4 `decoder` component, applies INT4 RTN, and saves a
   complete Hugging Face directory. Vision, audio, and embedding weights remain
   available for the later export.
2. `olive capture-onnx-graph --use_mobius_builder` loads that quantized
   directory and exports the four-component ORT GenAI package.

## Prerequisites

This recipe requires Olive multi-build support and the current Mobius
component/quantized-checkpoint integration. Until those changes are included in
published releases, install the tested source revisions and runtime
dependencies:

```bash
pip install "git+https://github.com/microsoft/Olive.git@4081c1bb"
pip install "git+https://github.com/onnxruntime/mobius.git@b9b4ef4"
pip install transformers torch onnxruntime-genai requests
```

Gemma 4 is gated. Accept the model license, then authenticate after installing
`huggingface_hub` through the dependencies above:

```bash
hf auth login
```

Run the commands below from this `multi_comp` directory.

## Step 1 — Quantize the decoder

```bash
olive run --config gemma4_quantize_then_export.json
```

The build selects only the decoder:

```json
{
    "components": ["decoder"],
    "pipeline": ["decoder_rtn"]
}
```

`Rtn` performs calibration-free INT4 weight quantization with group size 128.
The embedding table, LM head, and Gemma 4's runtime-specific
`per_layer_input_gate` / `per_layer_projection` modules remain floating point;
Mobius currently represents those two modules as ordinary Linear operators.
Olive saves a complete Hugging Face checkpoint, not a standalone decoder:

```text
gemma4_decoder_int4_hf/
  model/
    config.json
    generation_config.json
    model*.safetensors
    tokenizer and processor files
  model_config.json
  footprint.json
```

The complete directory is required because Mobius still needs the unquantized
vision encoder, audio encoder, and multimodal embedding components.

## Step 2 — Export all components with Mobius

```bash
olive capture-onnx-graph \
  --model_name_or_path gemma4_decoder_int4_hf/model \
  --use_mobius_builder \
  --trust_remote_code \
  --precision fp32 \
  --output_path exported_gemma4_int4_pkg
```

Mobius preserves the Olive-packed INT4 decoder weights and exports:

```text
exported_gemma4_int4_pkg/
  decoder/model.onnx
  vision_encoder/model.onnx
  audio_encoder/model.onnx
  embedding/model.onnx
  genai_config.json
  tokenizer.json
  processor and audio feature-extraction files
```

## Step 3 — Inference

Use the inference entry point in the parent Gemma 4 recipe:

```bash
python ../inference.py \
  --model-path exported_gemma4_int4_pkg \
  --prompt "What is the capital of France?" \
  --verbose
```

For CUDA inference, install `onnxruntime-genai-cuda` and change the export
precision to `fp16` on a CUDA-capable machine. The RTN stage itself can run on
CPU or CUDA.

## Notes

- `builds.components: ["decoder"]` scopes RTN to the language decoder while
  preserving the full Hugging Face checkpoint layout.
- `lm_head: false` and `embeds: false` avoid quantizing the tied/output tables.
- `modules_to_not_convert` keeps Gemma 4's per-layer input gate/projection in
  the floating-point format expected by the current Mobius graph.
- This is intentionally a quantize-then-export flow. The existing sibling
  recipes under `cpu/` and `cuda/` demonstrate export-then-ONNX-quantize flows.
- The model download and full quantization require substantial disk and memory.
