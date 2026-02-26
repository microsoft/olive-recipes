# Qwen2.5-VL-3B-Instruct ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

The pipeline exports three sub-models (vision encoder, text embedding, text decoder), applies graph optimizations (Cast chain elimination, Gemm→MatMul conversion), and quantizes all three sub-models to INT4.

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

### 1. Export & Optimize Models (CPU)

All graph transformations and quantization are declared in the JSON config files inside `cpu_and_mobile/`. The `optimize.py` script orchestrates the three Olive runs and generates the GenAI runtime configs.

```bash
cd cpu_and_mobile
```

| Command | Description |
|---------|-------------|
| `python optimize.py --device cpu` | Full pipeline: export, optimize, INT4 quantize (CPU) |
| `python optimize.py --device gpu` | Full pipeline with CUDA execution provider |
| `python optimize.py --skip-export` | Regenerate configs only (models already exported) |

> **Note:** The text model is always exported as INT4 via ModelBuilder. The vision encoder is graph-optimized and quantized to INT4 by Olive passes. The embedding model's Gather-based embedding table is quantized to INT4 using GatherBlockQuantized.
>
> The vision encoder is exported for a single image using the Dynamo exporter. At runtime, ONNX Runtime GenAI handles multiple images by calling the vision encoder once per image and concatenating the results — so there is no upper bound on the number of images passed to the model.

### 1b. Export CUDA Models (optional)

The `cuda/` directory contains configs that produce FP16 + INT4 models for CUDA. CPU models must be exported first (the CUDA export reuses the CPU `genai_config.json`).

```bash
cd cuda
python export.py
```

### 2. Run Inference

From the top-level model directory:

```bash
# Text-only (CPU models, default)
python inference.py --prompt "What is the capital of France?"

# With a single image
python inference.py --prompt "Describe this image" --image cat.jpeg

# CUDA models
python inference.py --model_path cuda/models --prompt "Describe this image" --image cat.jpeg

# Interactive mode
python inference.py --interactive
```

**Multi-image inference** is supported via `model-mm.py` from the `onnxruntime-genai` examples:

```bash
# Two images — compare or reason across multiple images
cd ../../onnxruntime-genai/examples/python
python model-mm.py -m ../../../olive-recipes/Qwen-Qwen2.5-VL-3B-Instruct/cpu_and_mobile/models -up "Are these two images the same?" --image_paths ../../../olive-recipes/Qwen-Qwen2.5-VL-3B-Instruct/cat.jpeg ../../../olive-recipes/Qwen-Qwen2.5-VL-3B-Instruct/cat.jpeg --non_interactive
```

## Evaluation

`eval.py` measures model quality on [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d) — a multiple-choice visual QA benchmark on scientific diagrams. It supports side-by-side comparison of the quantized ONNX model against the PyTorch FP32 baseline.

```bash
# ONNX only (fastest)
python eval.py --num_samples 100

# ONNX + PyTorch comparison
python eval.py --num_samples 100 --pytorch_model Qwen/Qwen2.5-VL-3B-Instruct

# Evaluate CUDA models
python eval.py --model_path cuda/models --num_samples 100
```

### Results (AI2D, 100 samples, CPU)

| Model | Accuracy | Avg latency |
|-------|----------|-------------|
| PyTorch FP32 (baseline) | 81.00% | 9.41 s/sample |
| **ONNX INT4 (quantized)** | **83.00%** | **7.76 s/sample** |
| Random chance | 25.00% | — |

- **Quantization accuracy delta: +2 pp** (81% → 83%)
- **Latency speedup: 1.21×** on CPU

> Results measured on CPU (Intel) with `--num_samples 100` from the AI2D test split. GPU results will differ.

## Directory Structure

```
Qwen-Qwen2.5-VL-3B-Instruct/
├── eval.py                    # AI2D accuracy evaluation (ONNX vs PyTorch)
├── inference.py               # ONNX Runtime GenAI inference
├── cat.jpeg                   # Sample test image
├── codes/                     # Custom Qwen2.5-VL PyTorch model adapted for ONNX export
├── cpu_and_mobile/
│   ├── optimize.py            # End-to-end Olive pipeline + GenAI config generation
│   ├── user_script.py         # Olive callbacks: model loading, dummy inputs, IO configs
│   ├── embedding.json         # Olive config: export → optimize → INT4
│   ├── vision.json            # Olive config: Dynamo export → graph surgeries → INT4
│   ├── text.json              # Olive config: ModelBuilder INT4
│   └── models/                # Exported ONNX models (generated)
└── cuda/
    ├── export.py              # CUDA export orchestrator (FP16 + INT4)
    ├── optimize.py            # Olive pipeline (same as CPU, uses CUDA configs)
    ├── user_script.py         # Olive callbacks
    ├── embedding.json         # Olive config with FP16 + INT4 + CUDA EP
    ├── vision.json            # Olive config with FP16 + INT4 + CUDA EP
    ├── text.json              # ModelBuilder INT4 with CUDA EP
    └── models/                # Exported CUDA ONNX models (generated)
```
