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

## Step 1 — Export with Mobius

```bash
olive capture-onnx-graph --model_name_or_path google/gemma-4-E2B-it --use_mobius_builder --precision fp32 --output_path gemma4_onnx
```

## Step 2 — Quantization

```bash
olive run --config gemma4_quantization.json
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
