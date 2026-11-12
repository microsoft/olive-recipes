# Qwen2.5-VL-3B-Instruct — Olive + Mobius Multi-Component Recipe

This recipe exports
[`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
with Olive's `MobiusBuilder`, then optimizes all three ONNX components from one
Olive multi-build config:

- `decoder`
- `vision_encoder`
- `embedding`

Mobius owns the model graph, weight mapping, ORT GenAI configuration, tokenizer,
and image processor generation. The target config loads the exported directory
as a `CompositeModel` and applies the configured pipeline to each component.

## Prerequisites

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI for the target:

| Target | Install command |
|---|---|
| CPU | `pip install onnxruntime-genai` |
| CUDA | `pip install onnxruntime-genai-cuda` |

Run commands from this `builtin` directory.

## Export and optimize

### CPU and mobile

Export the complete FP32 package:

```bash
olive capture-onnx-graph --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct --use_mobius_builder --trust_remote_code --precision fp32 --output_path cpu_and_mobile/mobius_base
```

Run all three component builds:

```bash
olive run --config cpu_and_mobile/config.json
```

`cpu_and_mobile/config.json` applies block-wise INT4 RTN to all three exported
components:

| Component | Pipeline |
|---|---|
| `decoder` | `OnnxBlockWiseRtnQuantization` |
| `vision_encoder` | `OnnxBlockWiseRtnQuantization` |
| `embedding` | `OnnxBlockWiseRtnQuantization` |

### CUDA

Export the complete FP16 package:

```bash
olive capture-onnx-graph \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --use_mobius_builder \
  --trust_remote_code \
  --precision fp16 \
  --output_path cuda/mobius_base
```

Run all three component builds:

```bash
olive run --config cuda/config.json
```

`cuda/config.json` applies block-wise INT4 RTN to all three exported components:

| Component | Pipeline |
|---|---|
| `decoder` | INT4 RTN |
| `vision_encoder` | INT4 RTN |
| `embedding` | INT4 RTN |

Both targets use the same two-stage flow:

1. `olive capture-onnx-graph` exports the complete three-component ORT GenAI
   package to `<config-dir>/mobius_base/`.
2. `olive run` executes the target's named component builds and automatically
   assembles the complete package under `<config-dir>/models/`, preserving the
   Mobius-generated runtime files and any components without a build.

The final ORT GenAI package uses Mobius's native component layout:

```text
cpu_and_mobile/models/
  decoder/model.onnx
  vision_encoder/model.onnx
  embedding/model.onnx
  genai_config.json
  processor_config.json
  tokenizer.json
  tokenizer_config.json
```

## Inference

```bash
# Text only
python inference.py --model_path cpu_and_mobile/models --prompt "What is the capital of France?"

# Image + text
python inference.py --model_path cpu_and_mobile/models --image cat.jpeg --prompt "Describe this image."

# CUDA package
python inference.py --model_path cuda/models --image cat.jpeg --prompt "Describe this image."
```

ORT GenAI executes the vision encoder only when images are present, fuses its
features in the embedding component, and runs autoregressive generation through
the decoder.

## Evaluation

`eval.py` evaluates the final package on AI2D:

```bash
python eval.py --model_path cpu_and_mobile/models --num_samples 100
python eval.py \
  --model_path cpu_and_mobile/models \
  --num_samples 100 \
  --pytorch_model Qwen/Qwen2.5-VL-3B-Instruct
```

Re-run evaluation when changing Mobius, quantization settings, or runtime
versions; results from the previous custom export graph are not comparable.

## Directory structure

```text
builtin/
  inference.py
  eval.py
  cat.jpeg
  cpu_and_mobile/
    config.json
  cuda/
    config.json
```
