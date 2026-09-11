# Qwen3.6-27B ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Qwen3.5 is a hybrid architecture combining GatedDeltaNet linear attention layers with standard full attention layers. The pipeline exports three sub-models (vision encoder, text embedding, text decoder) and applies graph optimizations. For CPU, all three sub-models are quantized to INT4. For CUDA, the vision encoder and embedding use FP16 while the text decoder uses INT4.

## Hardware Requirements

| | Min GPU Memory | Recommended |
|---|---|---|
| Export & optimize | ~52 GB | NVIDIA A100 80GB |
| ONNX inference (INT4) | ~5 GB | Any CUDA GPU with ≥8 GB |
| PyTorch inference (fp16) | ~52 GB | NVIDIA A100 80GB |

> **Note:** The model was exported and tested on an NVIDIA A100-SXM4-80GB. Export requires loading the full fp16 weights (~54 GB) into GPU memory.

## Prerequisites

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| CPU | `pip install onnxruntime-genai --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |
| GPU (CUDA) | `pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |

## Steps

### 1. Export & Optimize Models

**CPU (INT4 quantized):**

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu
```

**CUDA (FP16 vision/embedding + INT4 text decoder):**

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
cpu_and_mobile/models/          # or cuda/models/
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

# CUDA model
python inference.py --model_path cuda/models --prompt "Hello"
```

By default, `inference.py` loads from `cpu_and_mobile/models`. Pass `--model_path cuda/models` to run the CUDA build.

### 4. Evaluate on AI2D

Run the AI2D (diagram understanding) benchmark to measure accuracy:

```bash
# ONNX only (100 samples, default)
python eval.py

# Compare ONNX vs PyTorch side-by-side
python eval.py --pytorch_model Qwen/Qwen3.6-27B

# Larger evaluation
python eval.py --num_samples 500 --pytorch_model Qwen/Qwen3.6-27B
```

The eval script reports per-sample accuracy, average latency, and an ONNX-vs-PyTorch comparison summary.

### Benchmark Results (AI2D, 200 samples)

| | PyTorch (fp16) | ONNX (INT4) |
|---|---|---|
| Accuracy | 90.00% (180/200) | 86.00% (172/200) |
| Avg latency | 89.22s/sample | 26.99s/sample |
| GPU memory (avg) | 51.0 GB | 3.2 GB |
| GPU memory (peak) | 51.8 GB | 4.1 GB |

*Evaluated on NVIDIA A100-SXM4-80GB with system prompt `/no_think`.*
