# Gemma 4 E2B — Decoder KQuant + Vision RTN

This recipe uses two independent Olive component builds for
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it):

- `decoder`: PyTorch KQuant, asymmetric INT4, group size 32
- `vision_encoder`: PyTorch RTN, symmetric INT4, group size 128

Olive automatically assembles the component-only artifacts with the unchanged
audio and embedding weights into one standard Hugging Face checkpoint. Mobius
then loads that checkpoint through the ordinary
`olive capture-onnx-graph --use_mobius_builder` CLI.

## Prerequisites

```bash
pip install "git+https://github.com/microsoft/Olive.git@14bb7a6c"
pip install "git+https://github.com/onnxruntime/mobius.git@ea293cb"
pip install transformers torch onnxruntime-genai requests
hf auth login
```

Run the commands below from this `multi_comp` directory.

## Step 1 — Run and assemble both component builds

```bash
olive run --config gemma4_quantize_then_export.json
```

The config contains two disjoint builds under one shared output parent:

```json
{
    "builds": {
        "_default": {
            "output_dir": "gemma4_mixed_hf"
        },
        "decoder": {
            "components": ["decoder"],
            "pipeline": ["decoder_kquant"]
        },
        "vision": {
            "components": ["vision_encoder"],
            "pipeline": ["vision_rtn"]
        }
    }
}
```

Olive writes component-only shards for the optimized components and retains all
unbuilt tensors from the source checkpoint:

```text
gemma4_mixed_hf/
  config.json
  model.safetensors.index.json
  model-unoptimized-*.safetensors
  model_config.json
  decoder/
    component.json
    model-*.safetensors
  vision/
    component.json
    model-*.safetensors
```

The root is a standard HF checkpoint. Its `component_quantization` mapping
records the independent decoder and vision layouts. The LM head, embeddings,
audio encoder, and Gemma 4 `per_layer_input_gate` /
`per_layer_projection` modules remain floating point.

## Step 2 — Export with Mobius

```bash
olive capture-onnx-graph \
  --model_name_or_path gemma4_mixed_hf \
  --use_mobius_builder \
  --trust_remote_code \
  --precision fp32 \
  --output_path exported_gemma4_mixed_pkg
```

Output:

```text
exported_gemma4_mixed_pkg/
  decoder/model.onnx
  vision_encoder/model.onnx
  audio_encoder/model.onnx
  embedding/model.onnx
  genai_config.json
  tokenizer.json
  processor and audio feature-extraction files
```

The exported decoder contains 205 asymmetric group-32 `MatMulNBits` nodes. The
vision encoder contains 114 symmetric group-128 `MatMulNBits` nodes. Audio and
embedding remain floating point, and all 70 runtime-specific per-layer
gate/projection operations remain ordinary `MatMul`.

## Step 3 — Inference

Text:

```bash
python ../inference.py \
  --model-path exported_gemma4_mixed_pkg \
  --prompt "What is the capital of France?" \
  --verbose
```

Image:

```bash
python ../inference.py \
  --model-path exported_gemma4_mixed_pkg \
  --image path/to/image.jpg \
  --prompt "What animal is shown? Answer in one short sentence." \
  --verbose
```
