# Qwen3.5-27B ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI on CUDA.

Qwen3.5 is a hybrid architecture combining GatedDeltaNet linear attention layers with standard full attention layers. The pipeline exports three sub-models (vision encoder, text embedding, text decoder) and applies graph optimizations. The vision encoder and embedding use FP16 while the text decoder uses INT4.

## Prerequisites

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

```bash
pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple
```

## Steps

### 1. Export & Optimize Models

```bash
python optimize.py --config-dir cuda --device gpu
```

This runs three Olive pipelines:
- **embedding.json** — Exports the embedding fusion model (token embedding + image feature scatter)
- **vision.json** — Exports the vision encoder (packed patches → image features)
- **text.json** — Exports the text decoder via ModelBuilder (hybrid GatedDeltaNet + full attention, INT4)

Then generates `genai_config.json` and `processor_config.json` for the ORT GenAI runtime.

### 2. Output Structure

```
cuda/models/
├── embedding.onnx              # Embedding fusion model
├── embedding.onnx.data
├── vision.onnx                 # Vision encoder
├── vision.onnx.data
├── text.onnx                   # Text decoder (hybrid)
├── text.onnx.data
├── genai_config.json           # Runtime configuration
├── processor_config.json       # Image preprocessing
├── tokenizer.json
└── tokenizer_config.json
```

### 3. Run Inference

```bash
# Text-only
python inference.py --prompt "What is the capital of France?"

# Image + text
python inference.py --image photo.jpg --prompt "Describe this image"

# Interactive mode
python inference.py --interactive
```

### 4. Evaluate on AI2D

Run the AI2D (diagram understanding) benchmark to measure accuracy:

```bash
# ONNX only (100 samples, default)
python eval.py

# Compare ONNX vs PyTorch side-by-side
python eval.py --pytorch_model Qwen/Qwen3.5-27B

# Larger evaluation
python eval.py --num_samples 500 --pytorch_model Qwen/Qwen3.5-27B
```

The eval script reports per-sample accuracy, average latency, and an ONNX-vs-PyTorch comparison summary.
