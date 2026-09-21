# Gemma 4 E2B — Decoder KQuant + Vision RTN

This recipe optimizes for
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it):

## Prerequisites

```bash
pip install "git+https://github.com/microsoft/Olive.git"
pip install "git+https://github.com/onnxruntime/mobius.git"
pip install transformers torch onnxruntime-genai requests
hf auth login
```

Run the commands below from this `multi_comp` directory.

## Step 1 — Run and assemble both component builds

```bash
olive run --config gemma4_quantize.json
```

The config contains two disjoint builds under one shared output parent:

```json
{
    "builds": {
        "decoder": {
            "components": [ "decoder" ],
            "pipeline": [ "decoder_kquant" ]
        },
        "vision": {
            "components": [ "vision_encoder" ],
            "pipeline": [ "vision_rtn" ]
        },
        "embedding": {
            "components": ["embedding"],
            "pipeline": ["embedding_kquant"]
        }
    }
}
```

Olive writes component-only shards for the optimized components and retains all
unbuilt tensors from the source checkpoint:

```text
gemma4_quantized_hf/
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

## Step 2 — Export with Mobius

```bash
olive capture-onnx-graph --model_name_or_path gemma4_quantized_hf --use_mobius_builder --precision fp32 --output_path gemma4_onnx
```

Output:

```text
gemma4_onnx/
  decoder/model.onnx
  vision_encoder/model.onnx
  audio_encoder/model.onnx
  embedding/model.onnx
  genai_config.json
  tokenizer.json
  processor and audio feature-extraction files
```

## Step 3 — Inference

Text:

```bash
python ../inference.py --model-path gemma4_onnx --prompt "What is the capital of France?" --verbose
```

Image:

```bash
python ../inference.py --model-path gemma4_onnx --image ../cat.jpeg --prompt "What animal is shown? Answer in one short sentence." --verbose
```
