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

## Optimize

Run one command from this `multi_comp` directory:

```bash
python optimize.py --ep ov
python optimize.py --ep qnn
```

The script runs component quantization, Mobius ONNX export, and the selected
execution-provider optimization in sequence.

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
