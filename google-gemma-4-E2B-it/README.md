# Gemma 4 E2B (google/gemma-4-E2B-it)

Olive recipes for building and optimizing the Gemma 4 E2B multimodal model
using the [MobiusBuilder](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py) pass.

Gemma 4 is an any-to-any multimodal model with vision, audio, and text
capabilities. The pipeline produces four ONNX components (decoder,
vision_encoder, audio_encoder, embedding) for use with ORT GenAI.

## Prerequisites

```bash
pip install olive-ai[gpu,mobius-ai]
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| CPU | `pip install onnxruntime-genai` |
| GPU (CUDA) | `pip install onnxruntime-genai-cuda` |

## Steps

### 1. Export & Optimize Models

**CPU (FP32):**

```bash
python optimize.py --device cpu
```

**CUDA (FP16):**

```bash
python optimize.py --device gpu --variant fp16
```

**CUDA (INT4 quantized):**

```bash
python optimize.py --device gpu --variant int4
```

### 2. Output Structure

```
cpu/models/                     # or cuda/fp16/models/ or cuda/int4/models/
├── decoder/model.onnx          # Text decoder
├── vision_encoder/model.onnx   # Vision encoder
├── audio_encoder/model.onnx    # Audio encoder
├── embedding/model.onnx        # Embedding fusion
├── genai_config.json           # Runtime configuration
├── image_processor.json        # Vision preprocessing
├── audio_feature_extraction.json
├── tokenizer.json
└── tokenizer_config.json
```

### 3. Run Inference

```bash
# Text-only (CPU)
python inference.py --prompt "What is the capital of France?"

# CUDA INT4
python inference.py --device gpu --variant int4 --prompt "Explain quantum computing"

# Interactive mode
python inference.py --device gpu --variant fp16 --interactive

# Custom model path
python inference.py --model-path /path/to/models --prompt "Hello"
```

### 4. Evaluate (MMLU Pro)

```bash
# Quick eval (100 samples, CPU)
python eval.py

# Full eval on CUDA INT4
python eval.py --device gpu --variant int4 --limit 0

# Custom model path
python eval.py --model-path /path/to/models --task leaderboard_mmlu_pro
```

## Recipes

| Recipe | Device | Precision | Quantization |
|--------|--------|-----------|-------------|
| `cpu/config.json` | CPU | FP32 | None |
| `cuda/fp16/config.json` | CUDA | FP16 | None |
| `cuda/int4/config.json` | CUDA | FP16 → INT4 | Block-wise RTN (128, symmetric) |

## Model Info

- **Architecture**: Gemma 4 any-to-any multimodal (vision + audio + text)
- **Components**: decoder, vision_encoder, audio_encoder, embedding
- **Builder**: MobiusBuilder (builds all components in one pass)
- **HuggingFace**: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
