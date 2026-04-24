# Fara-7B ONNX Runtime GenAI Example

This example demonstrates how to convert [Fara-7B](https://huggingface.co/microsoft/Fara-7B) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Fara-7B is Microsoft's agentic small language model designed for computer use, based on Qwen 2.5 VL (7B). The pipeline exports three sub-models (vision encoder, text embedding, text decoder), applies graph optimizations, and quantizes all three sub-models.

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

### 1. Export & Optimize Models

All graph transformations and quantization are declared in the JSON config files inside `cpu_and_mobile/` and `cuda/`. The top-level `optimize.py` script orchestrates the three Olive runs and generates the GenAI runtime configs.

| Command | Description |
|---------|-------------|
| `python optimize.py --config-dir cpu_and_mobile --device cpu` | Full pipeline: export, optimize, INT4 quantize (CPU) |
| `python optimize.py --config-dir cuda --device gpu` | Full pipeline with FP16 + INT4 (CUDA) |
| `python optimize.py --config-dir cpu_and_mobile --skip-export` | Regenerate configs only (models already exported) |

> **Note:** The text model is always exported as INT4 via ModelBuilder. The vision encoder is graph-optimized and quantized by Olive passes. The embedding model's Gather-based embedding table is quantized using GatherBlockQuantized.
>
> The vision encoder is exported with dynamic `num_images` and `num_patches` dimensions using the Dynamo exporter, so a single ONNX model handles any number of images at any resolution in one call.

### 2. Run Inference

From the top-level model directory:

```bash
# Text-only (CPU models, default)
python inference.py --prompt "What is the capital of France?"

# With a single image
python inference.py --prompt "Describe this image" --image screenshot.png

# CUDA models
python inference.py --model_path cuda/models --prompt "Describe this image" --image screenshot.png

# Interactive mode
python inference.py --interactive
```

## Directory Structure

```
microsoft-Fara-7B/
└── builtin/
    ├── optimize.py                # End-to-end Olive pipeline + GenAI config generation
    ├── user_script.py             # Olive callbacks: model loading, dummy inputs, IO configs
    ├── inference.py               # ONNX Runtime GenAI inference
    ├── codes/                     # Custom Qwen2.5-VL PyTorch model adapted for ONNX export
    ├── cpu_and_mobile/
    │   ├── embedding.json         # Olive config: export → optimize → INT4
    │   ├── vision.json            # Olive config: Dynamo export → graph surgeries → INT4
    │   ├── text.json              # Olive config: ModelBuilder INT4
    │   └── models/                # Exported ONNX models (generated)
    └── cuda/
        ├── embedding.json         # Olive config with FP16 + CUDA EP
        ├── vision.json            # Olive config with FP16 + CUDA EP
        ├── text.json              # ModelBuilder INT4 with CUDA EP
        └── models/                # Exported CUDA ONNX models (generated)
```
