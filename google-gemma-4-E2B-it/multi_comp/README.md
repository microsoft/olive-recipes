# Gemma 4 E2B — Decoder KQuant + Vision RTN

This recipe optimizes for
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it):

## Prerequisites

```bash
pip install "git+https://github.com/microsoft/Olive.git"
pip install "git+https://github.com/onnxruntime/mobius.git"
pip install onnxruntime-genai requests
hf auth login
```

Run the commands below from this `multi_comp` directory.

## Step 1 — Export with Mobius

```bash
olive capture-onnx-graph --model_name_or_path google/gemma-4-E2B-it --use_mobius_builder --precision fp32 --output_path gemma4_onnx
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
olive run --config gemma4_quantization.json
```

## Step 3 — Inference

Text:

```bash
python ../inference.py --model-path gemma4_quantized_onnx --prompt "What is the capital of France?" --verbose
```

Image:

```bash
python ../inference.py --model-path gemma4_quantized_onnx --image ../cat.jpeg --prompt "What animal is shown? Answer in one short sentence." --verbose
```
