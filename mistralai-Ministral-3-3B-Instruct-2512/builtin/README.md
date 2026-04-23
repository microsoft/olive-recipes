# Ministral-3-3B ONNX Runtime GenAI Example

This example demonstrates how to convert [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Ministral-3-3B is a multimodal (VLM) model combining a Pixtral vision encoder with a Mistral text decoder using YaRN RoPE for extended context. The pipeline exports three sub-models:
- **Vision encoder** and **embedding** via [mobius](https://github.com/onnxruntime/mobius) (declarative ONNX graph construction); vision INT8-quantized via Olive
- **Text decoder** via Olive/ModelBuilder (GQA + k_quant_mixed INT4 quantization)

## Exported Configurations

| Component | CUDA | CPU |
|-----------|------|-----|
| Text decoder | k_quant_mixed INT4 (`MatMulNBits`) | k_quant_mixed INT4 (`MatMulNBits`) |
| Vision encoder | INT8 RTN (`MatMul8Bits`) | INT8 RTN (`MatMul8Bits`) |
| Embedding | FP16 | FP32 |

- **CUDA**: k_quant_mixed INT4 text decoder + INT8 vision + FP16 embedding. Optimized for throughput on NVIDIA GPUs.
- **CPU**: k_quant_mixed INT4 text decoder + INT8 vision + FP32 embedding. Uses FP32 for embedding (CPU EP promotes FP16 to FP32).

## Benchmark Results

Evaluated on [AI2D](https://allenai.org/data/diagrams) (science diagram multiple-choice QA, 4 options per question).

| Configuration | Accuracy | Samples | Model Size | Latency (s/sample) |
|---------------|----------|---------|------------|---------------------|
| PyTorch FP16 (CUDA) | 76.40% | 500 | ~7G | 0.14 |
| PyTorch FP32 (CPU) | 78.00% | 100 | ~7G | 21.66 |
| ONNX CUDA (k_quant_mixed + INT8 vision) | 74.00% | 500 | 3.83G | 0.14 |
| ONNX CPU (k_quant_mixed + INT8 vision) | 77.00% | 100 | 4.92G | 5.85 |

ONNX CUDA is within 2.4pp of PyTorch FP16 at **half the model size** (3.83G vs ~7G).
ONNX CPU achieves near-parity with PyTorch CPU at **70% smaller** (4.92G vs ~7G) and **3.7× faster**.

> **Latency Measurement:** Per-sample end-to-end inference time (image in → text out). Includes image preprocessing, tokenization, vision encoding, text generation (max 8 tokens), and decoding. Excludes model loading (one-time cost). Measured with `time.perf_counter()` averaged over all samples. No warmup run.

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

**CPU (k_quant_mixed INT4 text + INT8 vision + FP32 embedding):**

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu --dtype f32
```

**CUDA (k_quant_mixed INT4 text + INT8 vision + FP16 embedding):**

```bash
python optimize.py --config-dir cuda --device gpu
```

**With local dequantized checkpoint (skips FP8 dequant):**

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu --model-path /path/to/Ministral-3-3B-dequantized
```

This runs:
- **Olive/ModelBuilder** for text decoder (GQA attention, YaRN RoPE, k_quant_mixed INT4)
- **Mobius** for vision encoder (Pixtral, dynamic H×W, 2D RoPE) and embedding (token + image fusion)
- **Olive INT8 quantization** on vision encoder (both CUDA and CPU)

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

Run the AI2D science diagram QA benchmark (see [Benchmark Results](#benchmark-results) for expected accuracy):

```bash
# ONNX only (CPU)
python eval.py --device cpu --model_path cpu_and_mobile/models

# ONNX only (CUDA)
python eval.py --device cuda --model_path cuda/models

# PyTorch baseline (BF16 variant avoids FP8 kernel requirement)
python eval.py --skip_onnx --pytorch_model mistralai/Ministral-3-3B-Instruct-2512-BF16 --device cpu --num_samples 100

# Compare ONNX vs PyTorch side-by-side
python eval.py --model_path cuda/models --pytorch_model mistralai/Ministral-3-3B-Instruct-2512-BF16 --num_samples 100
```

> **Note:** The default HuggingFace checkpoint (`Ministral-3-3B-Instruct-2512`) uses FP8 weights,
> which require a specific CUDA kernel build. Use the `-BF16` variant for PyTorch baselines.

## Directory Structure

```
mistralai-Ministral-3-3B-Instruct-2512/builtin/
├── cpu_and_mobile/
│   ├── text.json               # k_quant_mixed INT4 text decoder config (Olive/ModelBuilder)
│   └── vision.json             # INT8 vision quantization (Olive, post-mobius)
├── cuda/
│   ├── text.json               # k_quant_mixed INT4 text decoder config (Olive/ModelBuilder)
│   └── vision.json             # INT8 vision quantization (Olive, post-mobius)
├── optimize.py                 # Export orchestrator (Olive + Mobius)
├── inference.py                # ORT GenAI inference (text + VLM)
├── eval.py                     # AI2D benchmark evaluation
├── requirements.txt
├── info.yml
└── README.md
```

> **Note:** Unlike Qwen VLM recipes (which use Olive for all 3 sub-models end-to-end),
> Ministral uses **mobius** for vision and embedding ONNX export, then **Olive** for
> INT8 quantization of vision. Embedding stays FP16 (or FP32 for CPU).

## Differences from Qwen VLM Recipes

Qwen VLM recipes export all three sub-models through Olive using JSON configs
(`text.json`, `vision.json`). Each JSON defines a multi-pass
pipeline: PyTorch export → graph surgery → ORT fusion → quantization/FP16.

This recipe takes a different approach for **vision and embedding**:

| Component | Qwen | Ministral | Why |
|-----------|------|-----------|-----|
| Text decoder | Olive/ModelBuilder (`text.json`) | Olive/ModelBuilder (`text.json`) | Same — ModelBuilder handles GQA + quantization |
| Vision encoder | Olive: PyTorch export + 5-6 passes | **Mobius** export + Olive INT8 (`vision.json`) | Pixtral's dynamic image dims break `torch.onnx.export` |
| Embedding | Olive: PyTorch export + 5 passes | **Mobius** export (FP16/FP32, no quantization) | Olive's GatherBlockQuantized has data format bugs |

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

**What Olive still handles:** `vision.json` applies
`OnnxBlockWiseRtnQuantization` (INT8) to the mobius-exported FP16 vision model
for both CUDA and CPU targets.

**Why optimize.py has more lines (~400) than Qwen (~170):**

| Code section | Lines | Why it can't be JSON-driven |
|---|---|---|
| `export_vision_and_embedding()` | ~55 | Olive has no mobius integration; Pixtral's dynamic dims cause dynamo failures |
| `update_genai_config()` | ~150 | Olive generates decoder config only; VLM 3-model config + transforms-based processor_config has no Olive pass |
| `quantize_vision_and_embedding()` | ~25 | Post-export INT8 on pre-built ONNX (Olive JSON-driven, but needs orchestration) |
| `fix_tokenizer()` | ~15 | No Olive tokenizer patching pass |

The text decoder export (`text.json`) and INT8 quantization (`vision.json`) ARE Olive JSON-driven — identical to Qwen.

## Known Limitations

- **CPU vision: language drift on some images.** The quantized vision encoder occasionally produces embeddings that cause the text decoder to respond in the wrong language (e.g., Chinese instead of English). This has been observed on specific test images and is a known artifact of vision quantization. INT8 significantly reduces this compared to INT4.
- **FP8 checkpoint requires special kernels.** The default HuggingFace checkpoint uses FP8 weights. Use the `-BF16` variant for PyTorch evaluation on machines without FP8 kernel support.
- **Multi-image supported.** The runtime supports variable-count multi-image inputs via PixtralImageSizes metadata. Requires onnxruntime-extensions ≥ PR #1050 and models exported with PixtralImageSizes in `processor_config.json`.

## Notes

- **CPU pipeline**: Mobius exports FP16 as an intermediate format. Olive then quantizes vision to INT8. For CPU deployment, use `--dtype f32` so embedding outputs float32 natively (CPU EP promotes FP16 to FP32, which causes genai dtype mismatches).
- **CUDA pipeline**: Mobius exports FP16 directly for vision/embedding. Olive quantizes vision to INT8. Text decoder uses k_quant_mixed INT4 via ModelBuilder.
- The HuggingFace checkpoint uses FP8 quantized weights. The export pipeline dequantizes these automatically (`weight * weight_scale_inv`).
- The tokenizer uses `TokenizersBackend` class which genai doesn't support. The optimize script fixes this to `LlamaTokenizer`.
- Pixtral vision supports dynamic image sizes (multiples of 28, up to 1540×1540).
- The text decoder includes `llama_4_attn_scale` for long-context attention (>16K tokens).
