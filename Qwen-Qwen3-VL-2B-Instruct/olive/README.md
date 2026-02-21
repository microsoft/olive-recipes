# Qwen3-VL-2B-Instruct ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

The pipeline exports three sub-models (vision encoder, text embedding, text decoder), applies graph optimizations, and optionally quantizes the vision and embedding models to INT4.

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

The `--device` flag determines which execution provider will be used at inference time.
The `--quantize` flag applies post-export quantization to the vision and embedding models.

| Command | Description |
|---------|-------------|
| `python optimize.py --device cpu --quantize` | Full pipeline: export, optimize, INT4 quantize (CPU) |
| `python optimize.py --device gpu --quantize` | Full pipeline with CUDA execution provider |
| `python optimize.py --device cpu` | Export and optimize only (no quantization) |
| `python optimize.py --quantize --skip-export` | Quantize only (models already exported) |

> **Note:** The text model is exported as INT4 via ModelBuilder. The vision and embedding models are exported as FP32, then graph-optimized and quantized to INT4 by default.
>
> The vision encoder is exported for a single image using the Dynamo exporter. At runtime, ONNX Runtime GenAI handles multiple images by calling the vision encoder once per image and concatenating the results — so there is no upper bound on the number of images passed to the model.

### 2. Run Inference

```bash
# Text-only
python inference.py --prompt "What is the capital of France?"

# With a single image
python inference.py --prompt "Describe this image" --image cat.jpeg

# Interactive mode
python inference.py --interactive
```

**Multi-image inference** is supported via `model-mm.py` from the `onnxruntime-genai` examples:

```bash
# Single image
python model-mm.py -m models -up "Describe this image" --image_paths cat.jpeg --non_interactive

# Two images — compare or reason across multiple images
python model-mm.py -m models -up "Are these two images the same?" --image_paths img1.jpeg img2.jpeg --non_interactive

# Three or more images
python model-mm.py -m models -up "Summarize all images" --image_paths img1.jpeg img2.jpeg img3.jpeg --non_interactive
```

## Evaluation

`eval.py` measures model quality on [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d) — a multiple-choice visual QA benchmark on scientific diagrams. It supports side-by-side comparison of the quantized ONNX model against the PyTorch FP32 baseline.

```bash
# ONNX only (fastest)
python eval.py --num_samples 100

# ONNX + PyTorch comparison
python eval.py --num_samples 100 --pytorch_model Qwen/Qwen3-VL-2B-Instruct
```

### Results (AI2D, 100 samples, CPU)

| Model | Accuracy | Avg latency |
|-------|----------|-------------|
| PyTorch FP32 (baseline) | 74.00% | 5.68 s/sample |
| **ONNX INT4 (quantized)** | **70.00%** | **3.92 s/sample** |
| Random chance | 25.00% | — |

- **Quantization accuracy loss: −4 pp** (74% → 70%)
- **Latency speedup: 1.45×** on CPU
- A system prompt forcing single-digit responses is applied by default (see `DEFAULT_SYSTEM_PROMPT` in `eval.py`). Without it, the ONNX model tends to produce verbose chain-of-thought answers that reduce accuracy by a further ~11 pp — a prompt-engineering artifact, not a model quality issue.

> Results measured on CPU (Intel) with `--num_samples 100` from the AI2D test split. GPU results will differ.

## File Structure

| File | Description |
|------|-------------|
| `optimize.py` | End-to-end pipeline: Olive export, config generation, graph optimization, quantization |
| `user_script.py` | Olive callbacks: model loading, dummy inputs, IO configs (referenced by JSON configs) |
| `inference.py` | ONNX Runtime GenAI inference (single prompt and interactive mode) |
| `eval.py` | Accuracy evaluation on AI2D benchmark (ONNX vs PyTorch comparison) |
| `embedding.json` | Olive config for embedding model export |
| `vision.json` | Olive config for vision encoder export (uses Dynamo exporter for correctness with dynamic control flow) |
| `text.json` | Olive config for text decoder export (ModelBuilder, INT4) |
| `codes/` | Custom Qwen3-VL PyTorch model adapted for ONNX export |
