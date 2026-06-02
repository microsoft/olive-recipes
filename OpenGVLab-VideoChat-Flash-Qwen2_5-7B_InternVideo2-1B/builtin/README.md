# VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B ONNX Runtime GenAI Example

This example demonstrates how to convert [VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B) — a vision-language model that pairs the InternVideo2-1B vision encoder with a Qwen2.5-7B decoder — to ONNX format using Olive and run inference with ONNX Runtime GenAI.

The pipeline exports three sub-models and assembles them into a GenAI runtime bundle:

| Sub-model | Source | Precision | Notes |
|---|---|---|---|
| `vision.onnx` | InternVideo2-1B + ToMe16 projector | FP32 | Image or video mode (`--mode`) |
| `embedding.onnx` | `embed_tokens` + visual feature merge | FP32 | ONNX-compatible `cumsum + where` merge at `<\|image_pad\|>` positions |
| `text.onnx` | Qwen2.5-7B decoder | INT4 | Quantized via `ModelBuilder` |

> This recipe currently targets **CPU only**. Vision and embedding sub-models are exported in FP32; only the text decoder is quantized.

## Prerequisites

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

```bash
pip install onnxruntime-genai
```

If the Olive cache is empty, the text export will pull weights from the Hugging Face Hub. Authenticate first:

```bash
huggingface-cli login
```

## Steps

### 1. Export & Optimize Models

All graph transformations and quantization are declared in the JSON config files inside `cpu_and_mobile/`. The top-level `optimize.py` script orchestrates the three Olive runs (each in its own subprocess so memory is fully released between exports) and then writes the GenAI runtime configs.

| Command | Description |
|---------|-------------|
| `python optimize.py --config-dir cpu_and_mobile --device cpu` | Full pipeline (image-mode vision, default) |
| `python optimize.py --config-dir cpu_and_mobile --device cpu --mode video` | Use 4-frame video-mode vision encoder |
| `python optimize.py --config-dir cpu_and_mobile --device cpu --staging-dir D:/staging` | Stage intermediate artifacts on a separate drive |
| `python optimize.py --config-dir cpu_and_mobile --device cpu --skip-export` | Re-run only the GenAI config generation |

The full HuggingFace model is ~15 GB; expect the export to take significant disk and memory.

### 2. Run Inference

From the `builtin/` directory:

```bash
# Text-only
python inference.py --prompt "What is the capital of France?"

# With an image
python inference.py --image photo.jpg --prompt "Describe this image"

# Custom model directory
python inference.py --model-dir cpu_and_mobile/models --image cat.png --prompt "What animal is this?"

# Interactive mode (type "image:/path/to/file.jpg your prompt" to attach an image)
python inference.py --interactive
```

## Directory Structure

```
OpenGVLab-VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B/
└── builtin/
    ├── optimize.py                # End-to-end Olive pipeline + GenAI config generation
    ├── user_script.py             # Olive callbacks: model loading, dummy inputs, IO configs
    ├── inference.py               # ONNX Runtime GenAI inference
    ├── info.yml                   # Recipe metadata
    ├── requirements.txt
    └── cpu_and_mobile/
        ├── embedding.json         # Olive config: export embedding + visual-merge head (FP32)
        ├── vision.json            # Olive config: 4-frame video vision encoder (FP32)
        ├── vision_image.json      # Olive config: single-image vision encoder (FP32)
        ├── text.json              # Olive config: ModelBuilder INT4 text decoder
        └── models/                # Exported ONNX models (generated)
```
