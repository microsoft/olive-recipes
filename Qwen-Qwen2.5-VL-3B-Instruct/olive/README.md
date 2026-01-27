# Qwen2.5-VL-3B-Instruct ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

## Prerequisites

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI based on your target device:

| Device | Install Command |
|--------|-----------------|
| GPU (CUDA) | `pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |
| CPU | `pip install onnxruntime-genai --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |

## Steps

### 1. Optimize Models

The `--device` flag determines which execution provider will be used at inference time.

| Device | Command | Description |
|--------|---------|-------------|
| GPU (default) | `python optimize.py` | Uses CUDA execution provider |
| CPU | `python optimize.py --device cpu` | Uses CPU execution provider |

> **Note:** The model is always exported in fp16 precision.

### 2. Run Inference

```bash
# Text-only
python inference.py --prompt "What is the capital of France?"

# With image
python inference.py --prompt "Describe this image" --image cat.jpeg

# Interactive mode
python inference.py --interactive
```
