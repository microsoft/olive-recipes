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

## Step 3 — Optimize for the target execution provider

OpenVINO:

```bash
olive run --config ov.json
```

QNN:

```bash
olive run --config qnn.json
```

## Run all three steps

Use `optimize.py` to run quantization, Mobius export, and target-specific optimization in sequence:

```bash
python optimize.py --ep OpenVINOExecutionProvider
python optimize.py --ep QNNExecutionProvider
```

The shorter `ov` and `qnn` aliases are also accepted.

## Inference

Use `gemma4_quantized_onnx` for OpenVINO and `gemma4_qnn` for QNN.

Text:

```bash
python ../inference.py --model-path <output-dir> --prompt "What is the capital of France?" --verbose
```

Image:

```bash
python ../inference.py --model-path <output-dir> --image ../cat.jpeg --prompt "What animal is shown? Answer in one short sentence." --verbose
```
