# Qwen3-VL-4B-Instruct ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

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

All graph transformations and quantization are declared in the JSON config files inside `cpu_and_mobile/` and `cuda/`. The top-level `optimize.py` script orchestrates the three Olive runs and generates the GenAI runtime configs.

| Command | Description |
|---------|-------------|
| `python optimize.py --config-dir cpu_and_mobile --device cpu` | Full pipeline: export, optimize, INT4 quantize (CPU) |
| `python optimize.py --config-dir cuda --device gpu` | Full pipeline with FP16 + INT4 (CUDA) |
| `python optimize.py --config-dir cpu_and_mobile --skip-export` | Regenerate configs only (models already exported) |

> **Note:** The text model is always exported as INT4 via ModelBuilder. The vision encoder is graph-optimized and quantized to INT4 by Olive passes. The embedding model's Gather-based embedding table is quantized to INT4 using GatherBlockQuantized.
>
> The vision encoder is exported for a single image using the Dynamo exporter. At runtime, ONNX Runtime GenAI handles multiple images by calling the vision encoder once per image and concatenating the results — so there is no upper bound on the number of images passed to the model.

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
# Adjust paths to your onnxruntime-genai checkout and model directory
python <onnxruntime-genai>/examples/python/model-mm.py \
    -m <path-to-builtin>/cpu_and_mobile/models \
    -up "Are these two images the same?" \
    --image_paths image1.jpeg image2.jpeg \
    --non_interactive
```

## Evaluation

`eval.py` measures model quality on [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d) — a multiple-choice visual QA benchmark on scientific diagrams. It supports side-by-side comparison of the quantized ONNX model against the PyTorch FP32 baseline.

```bash
# ONNX only (fastest)
python eval.py --num_samples 100

# ONNX + PyTorch comparison
python eval.py --num_samples 100 --pytorch_model Qwen/Qwen3-VL-4B-Instruct

# Evaluate CUDA models
python eval.py --model_path cuda/models --num_samples 100
```

### Results

#### CPU (AI2D, 100 samples)

| Model | Accuracy | Avg latency |
|-------|----------|-------------|
| PyTorch FP32 (baseline) | 83.00% | 10.09 s/sample |
| **ONNX INT4 (CPU)** | **83.00%** | **7.13 s/sample** |
| Random chance | 25.00% | — |

- **CPU INT4 accuracy delta: 0 pp** (83% → 83%)
- **CPU speedup: 1.41×** vs PyTorch FP32

#### CUDA (AI2D, 200 samples)

| Model | Accuracy | Avg latency |
|-------|----------|-------------|
| PyTorch FP32 (baseline) | 83.00% | 0.22 s/sample |
| **ONNX INT4+FP16 (CUDA)** | **81.00%** | **0.17 s/sample** |
| Random chance | 25.00% | — |

- **CUDA accuracy delta: −2 pp** (83% → 81%)
- **CUDA speedup: 1.29×** vs PyTorch FP32

A system prompt forcing single-digit responses is applied by default (see `DEFAULT_SYSTEM_PROMPT` in `eval.py`). Without it, the model tends to produce verbose chain-of-thought answers that reduce measured accuracy — a prompt-engineering artifact, not a model quality issue.

> CPU results measured with `--num_samples 100`; CUDA results measured with `--num_samples 200` from the AI2D test split.

## Directory Structure

```
Qwen-Qwen3-VL-4B-Instruct/
├── LICENSE
└── builtin/
    ├── optimize.py                # End-to-end Olive pipeline + GenAI config generation
    ├── user_script.py             # Olive callbacks: model loading, dummy inputs, IO configs
    ├── eval.py                    # AI2D accuracy evaluation (ONNX vs PyTorch)
    ├── inference.py               # ONNX Runtime GenAI inference
    ├── cat.jpeg                   # Sample test image
    ├── codes/                     # Custom Qwen3-VL PyTorch model adapted for ONNX export
    ├── cpu_and_mobile/
    │   ├── embedding.json         # Olive config: export → optimize → INT4
    │   ├── vision.json            # Olive config: Dynamo export → graph surgeries → INT4
    │   ├── text.json              # Olive config: ModelBuilder INT4
    │   └── models/                # Exported ONNX models (generated)
    └── cuda/
        ├── embedding.json         # Olive config: export → optimize → FP16 + CUDA EP
        ├── vision.json            # Olive config: Dynamo export → graph surgeries → FP16 + CUDA EP
        ├── text.json              # ModelBuilder INT4 with CUDA EP
        └── models/                # Exported CUDA ONNX models (generated)
```
