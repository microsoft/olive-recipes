# Qwen3.5-4B ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Qwen3.5 is a hybrid architecture combining GatedDeltaNet linear attention layers with standard full attention layers. The pipeline exports three sub-models (vision encoder, text embedding, text decoder) and applies graph optimizations. For CPU, all three sub-models are quantized to INT4. For CUDA, the vision encoder and embedding use FP16 while the text decoder uses INT4.

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
# CPU
python -m onnxruntime_genai.models.model_mm -m cpu_and_mobile/models --max_length 4096

# CUDA
python -m onnxruntime_genai.models.model_mm -m cuda/models --max_length 4096
```
