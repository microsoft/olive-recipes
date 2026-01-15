# Qwen2.5-VL-3B-Instruct ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

## Prerequisites

```bash
pip install -r requirements.txt
pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple
```

## Steps

### 1. Optimize Models

```bash
python optimize.py
```

### 2. Run Inference

```bash
# Text-only
python inference.py --prompt "What is the capital of France?"

# With image
python inference.py --prompt "Describe this image" --image cat.jpg

# Interactive mode
python inference.py --interactive
```
