# Qwen2.5-VL-3B-Instruct — Olive + Mobius Multi-Component Recipe

This recipe exports
[`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
with Olive's `MobiusBuilder`, then optimizes all three ONNX components from one
Olive multi-build config:

- `decoder`
- `vision_encoder`
- `embedding`

Mobius owns the model graph, weight mapping, ORT GenAI configuration, tokenizer,
and image processor generation. The previous custom PyTorch model and three
independent component configs are no longer required.

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

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu
```

`cpu_and_mobile/config.json` is one Olive config containing three named builds.
All three components use block-wise INT4 RTN:

| Build | Pipeline |
|---|---|
| `decoder` | `OnnxBlockWiseRtnQuantization` |
| `vision_encoder` | `OnnxBlockWiseRtnQuantization` |
| `embedding` | `OnnxBlockWiseRtnQuantization` |

### CUDA

```bash
python optimize.py --config-dir cuda --device gpu
```

`cuda/config.json` preserves the previous target intent:

| Build | Pipeline |
|---|---|
| `decoder` | INT4 RTN |
| `vision_encoder` | Mobius FP16, Olive resave |
| `embedding` | Mobius FP16, Olive resave |

Both flows have two stages:

1. Olive runs `MobiusBuilder` once and saves the complete package under
   `<config-dir>/mobius_base/`.
2. Olive runs the target's single `config.json`; its `builds` select and
   optimize the three Mobius components into `<config-dir>/models/`.

To reuse an existing Mobius export while rerunning the three component builds:

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu --skip-export
```

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
python inference.py \
  --model_path cpu_and_mobile/models \
  --prompt "What is the capital of France?"

# Image + text
python inference.py \
  --model_path cpu_and_mobile/models \
  --image cat.jpeg \
  --prompt "Describe this image."

# CUDA package
python inference.py \
  --model_path cuda/models \
  --image cat.jpeg \
  --prompt "Describe this image."
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
  optimize.py
  inference.py
  eval.py
  cat.jpeg
  cpu_and_mobile/
    config.json
  cuda/
    config.json
```
