# Qwen3.6-35B-A3B MoE ONNX Runtime GenAI Example (CUDA)

This example demonstrates how to convert [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) Mixture-of-Experts vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI, optimized for the **CUDA execution provider** on consumer GPUs (e.g. RTX 4090, 24 GB).

Qwen3.6-35B-A3B is a hybrid MoE model with 256 experts (8 routed + 1 shared per layer), GatedDeltaNet linear attention, and a Qwen3.6 vision encoder. It reuses the `Qwen3_5MoeForConditionalGeneration` architecture. The pipeline exports three sub-models (vision encoder, text embedding, text decoder), applies graph optimizations, quantizes the text decoder to INT4 (QMoE), and runs the vision/embedding encoders in FP16.

## Prerequisites

```bash
pip install olive-ai onnxruntime-genai-cuda transformers torch safetensors
```

A CUDA-enabled ONNX Runtime GenAI build with the `QMoE`, `LinearAttention`, `CausalConvWithState`, and `GroupQueryAttention` contrib kernels is required.

## Steps

### 1. Export & Optimize Models (CUDA)

```bash
# INT4 (QMoE) text decoder + FP16 vision/embedding (recommended for RTX 4090)
python optimize.py --config-dir cuda --device gpu

# Regenerate configs only (models already exported)
python optimize.py --config-dir cuda --device gpu --skip-export
```

| Command | Description |
|---------|-------------|
| `python optimize.py --config-dir cuda --device gpu` | Full pipeline: export, optimize, INT4 quantize for CUDA |
| `python optimize.py --config-dir cuda --device gpu --skip-export` | Regenerate configs only (models already exported) |
| `python optimize.py --config-dir cuda --device gpu --context-length 8192` | Export with custom context length |
| `python optimize.py --config-dir cpu_and_mobile --device cpu` | CPU fallback build |

> **Note:** The text decoder is exported as INT4 via ModelBuilder with 256 MoE experts using QMoE symmetric blockwise quantization (CUDA `QMoE` kernel) with FP16 activations. The vision encoder and embedding model are exported and converted to FP16 via Olive's `OrtTransformersOptimization` pass with graph surgeries (PackedAttentionToLoopMHA, GemmToMatMulAdd) so their outputs match the FP16 decoder inputs.

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

Qwen3.6-35B-A3B is a `Qwen3_5MoeForConditionalGeneration` multimodal model with three components:

```
Image [B, C, H, W]
  |
  v
vision.onnx (Qwen3.6 vision encoder, FP16)
  |
  v  image_features [num_patches, hidden]
  |
  +--- input_ids [B, seq_len] ---> embedding.onnx (embed_tokens + scatter, FP16)
                                     |
                                     v  inputs_embeds [B, seq_len, 2048]
                                     |
                                     +---> text.onnx (40 hybrid layers: GatedDeltaNet + MoE, INT4 QMoE / FP16)
                                             |
                                             v  logits -> tokens
```

- **Vision**: Qwen3.6 vision encoder processes images into patch features.
- **Embedding**: Looks up token embeddings and scatters vision features into image-token positions.
- **Text**: 40 hybrid decoder layers alternating between GatedDeltaNet linear attention and full attention, each with a MoE MLP (256 experts, top-8 routing + shared expert with sigmoid gating).

## Directory Structure

```
Qwen-Qwen3.6-35B-A3B/
├── LICENSE
└── builtin/
    ├── optimize.py                # End-to-end Olive pipeline + GenAI config generation
    ├── user_script.py             # Olive callbacks: model loading, dummy inputs, IO configs
    ├── inference.py               # ONNX Runtime GenAI inference
    ├── info.yml                   # Recipe metadata
    ├── README.md
    ├── codes/
    │   ├── __init__.py
    │   └── modeling_qwen3_5_moe.py  # Custom ONNX-export-friendly MoE model (shared arch)
    ├── cuda/                       # CUDA execution provider configs (recommended)
    │   ├── text.json              # Olive config: ModelBuilder INT4 QMoE (CUDA)
    │   ├── embedding.json         # Olive config: OnnxConversion + FP16 + graph surgeries
    │   ├── vision.json            # Olive config: Dynamo export + FP16 + graph surgeries
    │   └── models/                # Exported ONNX models (generated)
    └── cpu_and_mobile/            # CPU configs (fallback)
        ├── text.json
        ├── embedding.json
        ├── vision.json
        └── models/
```
