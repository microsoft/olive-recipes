# Ministral-3-3B ONNX Runtime GenAI Example

This example demonstrates how to convert [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Ministral-3-3B is a multimodal (VLM) model combining a Pixtral vision encoder with a Mistral text decoder using YaRN RoPE for extended context. The pipeline exports three sub-models:
- **Vision encoder** and **embedding** via [mobius](https://github.com/onnxruntime/mobius) (declarative ONNX graph construction)
- **Text decoder** via Olive/ModelBuilder (GQA + INT4/FP16 quantization)

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

**CPU (INT4 text decoder, FP16 vision/embedding):**

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu
```

**CUDA (FP16):**

```bash
python optimize.py --config-dir cuda --device gpu
```

**With local dequantized checkpoint (skips FP8 dequant):**

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu --model-path /path/to/Ministral-3-3B-dequantized
```

This runs:
- **Olive/ModelBuilder** for text decoder (GQA attention, YaRN RoPE, INT4/FP16)
- **Mobius** for vision encoder (Pixtral, dynamic H×W, 2D RoPE) and embedding (token + image fusion)

Then generates `genai_config.json` and `processor_config.json` for the ORT GenAI runtime.

### 2. Output Structure

```
cpu_and_mobile/models/          # or cuda/models/
├── vision.onnx                 # Pixtral vision encoder
├── vision.onnx.data
├── embedding.onnx              # Embedding fusion model
├── embedding.onnx.data
├── text.onnx                   # Text decoder (Mistral + YaRN)
├── text.onnx.data
├── genai_config.json           # Runtime configuration
├── processor_config.json       # Pixtral image preprocessing
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

Alternatively, use the built-in GenAI multimodal demo:

```bash
python -m onnxruntime_genai.models.model_mm -m cpu_and_mobile/models --max_length 4096
```

## Notes

- The HuggingFace checkpoint uses FP8 quantized weights. The export pipeline dequantizes these automatically (`weight * weight_scale_inv`).
- The tokenizer uses `TokenizersBackend` class which genai doesn't support. The optimize script fixes this to `LlamaTokenizer`.
- Pixtral vision supports dynamic image sizes (multiples of 28, up to 1540×1540).
- The text decoder includes `llama_4_attn_scale` for long-context attention (>16K tokens).
