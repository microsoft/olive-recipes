# Qwen3.5-35B-A3B MoE ONNX Runtime GenAI Example

This example demonstrates how to convert [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) Mixture-of-Experts vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Qwen3.5-35B-A3B is a hybrid MoE model with 256 experts (8 routed + 1 shared per layer), GatedDeltaNet linear attention, and a Qwen3.5 vision encoder. The pipeline exports three sub-models (vision encoder, text embedding, text decoder), applies graph optimizations, and quantizes the text decoder to INT4.

## Prerequisites

```bash
pip install olive-ai onnxruntime-genai transformers torch safetensors
```

## Steps

### 1. Export & Optimize Models (CPU)

```bash
# INT4 text decoder + FP32 vision/embedding (recommended)
python optimize.py --config-dir cpu_and_mobile --device cpu

# Regenerate configs only (models already exported)
python optimize.py --config-dir cpu_and_mobile --device cpu --skip-export
```

| Command | Description |
|---------|-------------|
| `python optimize.py --config-dir cpu_and_mobile --device cpu` | Full pipeline: export, optimize, INT4 quantize |
| `python optimize.py --config-dir cpu_and_mobile --skip-export` | Regenerate configs only (models already exported) |
| `python optimize.py --config-dir cpu_and_mobile --context-length 8192` | Export with custom context length |

> **Note:** The text decoder is exported as INT4 via ModelBuilder with 256 MoE experts using QMoE symmetric blockwise quantization. The vision encoder and embedding model are exported as FP32 via Olive's OnnxConversion pass with graph surgeries (PackedAttentionToLoopMHA, GemmToMatMulAdd).

### 2. Run Inference

```bash
# Text-only prompt
python inference.py --prompt "What is 2+2?"

# With an image
python inference.py --prompt "Describe this image" --image photo.jpg

# Interactive mode
python inference.py --interactive
```

## Architecture

Qwen3.5-35B-A3B is a `Qwen3_5MoeForConditionalGeneration` multimodal model with three components:

```
Image [B, C, H, W]
  |
  v
vision.onnx (Qwen3.5 vision encoder)
  |
  v  image_features [num_patches, hidden]
  |
  +--- input_ids [B, seq_len] ---> embedding.onnx (embed_tokens + scatter)
                                     |
                                     v  inputs_embeds [B, seq_len, 2048]
                                     |
                                     +---> text.onnx (40 hybrid layers: GatedDeltaNet + MoE)
                                             |
                                             v  logits -> tokens
```

- **Vision**: Qwen3.5 vision encoder processes images into patch features.
- **Embedding**: Looks up token embeddings and scatters vision features into image-token positions.
- **Text**: 40 hybrid decoder layers alternating between GatedDeltaNet linear attention and full attention, each with a MoE MLP (256 experts, top-8 routing + shared expert with sigmoid gating).

## Directory Structure

```
Qwen-Qwen3.5-35B-A3B/
├── LICENSE
└── builtin/
    ├── optimize.py                # End-to-end Olive pipeline + GenAI config generation
    ├── user_script.py             # Olive callbacks: model loading, dummy inputs, IO configs
    ├── inference.py               # ONNX Runtime GenAI inference
    ├── info.yml                   # Recipe metadata
    ├── README.md
    ├── codes/
    │   ├── __init__.py
    │   └── modeling_qwen3_5_moe.py  # Custom ONNX-export-friendly MoE model
    └── cpu_and_mobile/
        ├── text.json              # Olive config: ModelBuilder INT4 (QMoE)
        ├── embedding.json         # Olive config: OnnxConversion + graph surgeries
        ├── vision.json            # Olive config: Dynamo export + graph surgeries
        └── models/                # Exported ONNX models (generated)
```
