# Ministral-3-3B ONNX Runtime GenAI Example

This example demonstrates how to convert [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Ministral-3-3B is a multimodal (VLM) model combining a Pixtral vision encoder with a Mistral text decoder using YaRN RoPE for extended context. The pipeline exports three sub-models:
- **Vision encoder** and **embedding** via [mobius](https://github.com/onnxruntime/mobius) (declarative ONNX graph construction); vision optionally INT4-quantized via Olive for CPU
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

**CPU (INT4 all models):**

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu
```

**CUDA (FP16 all models):**

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
- **Olive INT4 quantization** on vision (cpu_and_mobile only; embedding stays FP16)

Then generates `genai_config.json` and `processor_config.json` for the ORT GenAI runtime.

### 2. Output Structure

```
cpu_and_mobile/models/          # or cuda/models/
├── decoder/
│   ├── model.onnx              # Text decoder (Mistral + YaRN)
│   └── model.onnx.data
├── vision/
│   ├── model.onnx              # Pixtral vision encoder (FP16)
│   └── model.onnx.data
├── embedding/
│   ├── model.onnx              # Embedding fusion model (FP16)
│   └── model.onnx.data
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

### 4. Evaluate

Run the AI2D science diagram QA benchmark:

```bash
# ONNX only (CPU INT4)
python eval.py --device cpu --model_path cpu_and_mobile/models

# ONNX only (CUDA FP16)
python eval.py --device cuda --model_path cuda/models

# Compare ONNX vs PyTorch reference
python eval.py --pytorch_model mistralai/Ministral-3-3B-Instruct-2512 --num_samples 100
```

Expected precision gaps (ONNX vs PyTorch):
- **FP32**: ~0 pp (exact parity)
- **FP16**: <2 pp (precision loss)
- **INT4**: <5 pp (quantization loss)

## Directory Structure

```
mistralai-Ministral-3-3B-Instruct-2512/builtin/
├── cpu_and_mobile/
│   ├── text.json               # INT4 text decoder config (Olive/ModelBuilder)
│   └── vision.json             # INT4 vision quantization (Olive, post-mobius)
├── cuda/
│   └── text.json               # FP16 text decoder config (Olive/ModelBuilder)
├── optimize.py                 # Export orchestrator (Olive + Mobius)
├── inference.py                # ORT GenAI inference (text + VLM)
├── eval.py                     # AI2D benchmark evaluation
├── requirements.txt
├── info.yml
└── README.md
```

> **Note:** Unlike Qwen VLM recipes (which use Olive for all 3 sub-models end-to-end),
> Ministral uses **mobius** for vision and embedding ONNX export, then **Olive** for
> INT4 quantization (cpu_and_mobile only). The CUDA target uses FP16 from mobius directly.

## Differences from Qwen VLM Recipes

Qwen VLM recipes export all three sub-models through Olive using JSON configs
(`text.json`, `vision.json`, `embedding.json`). Each JSON defines a multi-pass
pipeline: PyTorch export → graph surgery → ORT fusion → quantization/FP16.

This recipe takes a different approach for **vision and embedding**:

| Component | Qwen | Ministral | Why |
|-----------|------|-----------|-----|
| Text decoder | Olive/ModelBuilder (`text.json`) | Olive/ModelBuilder (`text.json`) | Same — ModelBuilder handles GQA + quantization |
| Vision encoder | Olive: PyTorch export + 5-6 passes | **Mobius** export + Olive INT4 (`vision.json`) | Pixtral's dynamic image dims break `torch.onnx.export` |
| Embedding | Olive: PyTorch export + 5 passes | **Mobius** export (FP16, no INT4) | INT4 breaks embedding's Equal/Gather logic |

**Why does Ministral use mobius instead of Olive for export?** Mobius constructs
the ONNX graph declaratively rather than tracing through PyTorch. The resulting
models already contain the graph optimizations that Qwen's Olive passes spend
5-6 steps creating:

- **Fused operators:** `MultiHeadAttention`, `SkipSimplifiedLayerNormalization`,
  `RotaryEmbedding` — already present in mobius output (Qwen achieves these via
  `OrtTransformersOptimization`)
- **FP16 weights:** all 840M vision params exported as FP16 directly (Qwen
  converts from FP32 via `OnnxFloatToFloat16`)
- **Clean graph:** 0 Gemm nodes, 0 redundant Cast chains (Qwen cleans these
  via `GemmToMatMulAdd` and `OnnxPeepholeOptimizer`)
- **No PyTorch export artifacts:** no `PackedAttentionToLoopMHA` surgery needed
  since mobius doesn't go through dynamo

**What Olive still handles:** For `cpu_and_mobile`, `vision.json` applies
`OnnxBlockWiseRtnQuantization` (INT4) to the mobius-exported FP16 vision model.
For `cuda`, no additional Olive passes are needed — FP16 is optimal for GPU.

**Why optimize.py has more lines (~400) than Qwen (~170):**

| Code section | Lines | Why it can't be JSON-driven |
|---|---|---|
| `export_vision_and_embedding()` | ~55 | Olive has no mobius integration; Pixtral's dynamic dims cause dynamo failures |
| `update_genai_config()` | ~150 | Olive generates decoder config only; VLM 3-model config + transforms-based processor_config has no Olive pass |
| `quantize_vision_and_embedding()` | ~25 | Post-export INT4 on pre-built ONNX (Olive JSON-driven, but needs orchestration) |
| `fix_tokenizer()` | ~15 | No Olive tokenizer patching pass |

The text decoder export (`text.json`) and INT4 quantization (`vision.json`) ARE Olive JSON-driven — identical to Qwen.

## Notes

- The HuggingFace checkpoint uses FP8 quantized weights. The export pipeline dequantizes these automatically (`weight * weight_scale_inv`).
- The tokenizer uses `TokenizersBackend` class which genai doesn't support. The optimize script fixes this to `LlamaTokenizer`.
- Pixtral vision supports dynamic image sizes (multiples of 28, up to 1540×1540).
- The text decoder includes `llama_4_attn_scale` for long-context attention (>16K tokens).
