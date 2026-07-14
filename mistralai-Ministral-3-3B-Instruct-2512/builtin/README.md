# Ministral-3-3B ONNX Runtime GenAI Example

This example demonstrates how to convert [Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) vision-language model to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Ministral-3-3B is a multimodal (VLM) model combining a Pixtral vision encoder with a Mistral text decoder using YaRN RoPE for extended context. The pipeline exports three sub-models:
- **Vision encoder** and **embedding** via Olive/MobiusBuilder pass (`vision_embedding_export.json`); vision INT8-quantized via Olive
- **Text decoder** via Olive/ModelBuilder (GQA + k_quant_mixed INT4 quantization)

## Exported Configurations

| Component | CUDA | CPU | WebGPU |
|-----------|------|-----|--------|
| Text decoder | k_quant_mixed INT4 (`MatMulNBits`) | k_quant_mixed INT4 (`MatMulNBits`) | k_quant_mixed INT4 (`MatMulNBits`) |
| Vision encoder | INT8 RTN, asymmetric block 32 (`MatMulNBits`) | INT8 RTN, symmetric block 128 (`MatMulNBits`) | INT8 RTN, asymmetric block 32 (`MatMulNBits`) |
| Embedding | FP16 | FP32 | FP16 |

- **CUDA**: k_quant_mixed INT4 text decoder + asymmetric block-32 INT8 vision + FP16 embedding. Optimized for throughput on NVIDIA GPUs.
- **CPU**: k_quant_mixed INT4 text decoder + INT8 vision + FP32 embedding. Uses FP32 for embedding (CPU EP promotes FP16 to FP32).
- **WebGPU**: k_quant_mixed INT4 text decoder + asymmetric block-32 INT8 vision + FP16 embedding. Uses WebGPU provider options in `genai_config.json`.

## Benchmark Results

Evaluated on [AI2D](https://allenai.org/data/diagrams) (science diagram multiple-choice QA, 4 options per question).

| Configuration | Accuracy | Samples | Model Size | Latency (s/sample) |
|---------------|----------|---------|------------|---------------------|
| PyTorch FP16 (CUDA, BF16 checkpoint) | 74.20% (371/500) | 500 | N/A | 0.17 |
| ONNX CUDA INT4 text + INT8 vision (asym block 32, BF16 checkpoint) | 73.00% (365/500) | 500 | 3.86 GB | 0.11 |
| ONNX CPU INT4 text + INT8 vision (sym block 128, BF16 checkpoint) | 72.80% (364/500) | 500 | 4.92 GB | 8.05 |

The current CUDA ONNX export is faster than PyTorch on this benchmark and is **1.20pp lower** in accuracy. A vision-quantization sweep found that asymmetric block-32 INT8 vision preserves the Mobius FP16 vision features best: on the feature probe it matched FP16 vision cosine similarity (`0.864774` vs `0.864780`), and on 500 AI2D samples it reached 365/500 versus 367/500 for the same package with unquantized FP16 vision.

Export validation status:

| Target | Package Size | Validation |
|--------|--------------|------------|
| CPU | 4.92 GB | Exported and evaluated; decoder, embedding, and vision ONNX graphs load; no hard-linked external data files |
| CUDA | 3.86 GB | Exported and evaluated; decoder and vision ONNX graphs contain `MatMulNBits`; no hard-linked external data files |
| WebGPU | 3.86 GB | Exported; decoder, embedding, and vision ONNX graphs load; no hard-linked external data files |

> **Latency Measurement:** Per-sample end-to-end inference time (image in → text out). Includes image preprocessing, tokenization, vision encoding, text generation, and decoding. Answers are short (typically 1-2 tokens for multiple-choice). Excludes model loading (one-time cost). Measured with `time.perf_counter()` averaged over all samples. No warmup run.

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
python optimize.py --config-dir cpu_and_mobile --device cpu
```

**CUDA (k_quant_mixed INT4 text + INT8 vision + FP16 embedding):**

```bash
python optimize.py --config-dir cuda --device gpu
```

**WebGPU (k_quant_mixed INT4 text + INT8 vision + FP16 embedding):**

```bash
python optimize.py --config-dir webgpu --device webgpu
```

**With a local or alternate checkpoint:**

```bash
python optimize.py --config-dir cpu_and_mobile --device cpu --model-path /path/to/Ministral-3-3B-dequantized
```

This runs:
- **Olive/ModelBuilder** for text decoder (GQA attention, YaRN RoPE, k_quant_mixed INT4)
- **Olive/MobiusBuilder** (`vision_embedding_export.json`) for vision encoder (Pixtral, dynamic H×W, 2D RoPE) and embedding (token + image fusion)
- **Olive INT8 quantization** (`vision.json`) on vision encoder (CPU, CUDA, and WebGPU)

Then generates `genai_config.json` and `processor_config.json` for the ORT GenAI runtime.

### 2. Output Structure

```
cpu_and_mobile/models/          # or cuda/ or webgpu/models/
├── decoder/
│   ├── model.onnx              # Text decoder (Mistral + YaRN)
│   └── model.onnx.data
├── vision/
│   ├── model.onnx              # Pixtral vision encoder (INT8)
│   └── model.onnx.data
├── embedding/
│   ├── model.onnx              # Embedding fusion model (FP16/FP32)
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

> **Note:** This recipe uses the BF16 Hugging Face checkpoint by default. The FP8 checkpoint
> (`Ministral-3-3B-Instruct-2512`) can require CUDA kernels that are not available on all machines.

## Directory Structure

```
mistralai-Ministral-3-3B-Instruct-2512/builtin/
├── cpu_and_mobile/
│   ├── text.json                       # k_quant_mixed INT4 text decoder config (Olive/ModelBuilder)
│   ├── vision_embedding_export.json    # Vision+embedding export (Olive/MobiusBuilder, FP32)
│   └── vision.json                     # INT8 vision quantization (Olive)
├── cuda/
│   ├── text.json                       # k_quant_mixed INT4 text decoder config (Olive/ModelBuilder)
│   ├── vision_embedding_export.json    # Vision+embedding export (Olive/MobiusBuilder, FP16)
│   └── vision.json                     # INT8 vision quantization (Olive)
├── webgpu/
│   ├── text.json                       # k_quant_mixed INT4 text decoder config (Olive/ModelBuilder)
│   ├── vision_embedding_export.json    # Vision+embedding export (Olive/MobiusBuilder, FP16)
│   └── vision.json                     # INT8 vision quantization (Olive)
├── optimize.py                 # Export orchestrator (all-Olive pipeline)
├── inference.py                # ORT GenAI inference (text + VLM)
├── eval.py                     # AI2D benchmark evaluation
├── requirements.txt
├── info.yml
└── README.md
```

> **Note:** Unlike Qwen VLM recipes (which use Olive for all 3 sub-models end-to-end via PyTorch export),
> Ministral uses the **Olive MobiusBuilder pass** (`vision_embedding_export.json`) for vision and embedding
> ONNX export, then **Olive INT8 quantization** (`vision.json`) for vision.
> Embedding stays FP16 (gpu/webgpu) or FP32 (cpu_and_mobile).

## Differences from Qwen VLM Recipes

Qwen VLM recipes export all three sub-models through Olive using JSON configs
(`text.json`, `vision.json`). Each JSON defines a multi-pass
pipeline: PyTorch export → graph surgery → ORT fusion → quantization/FP16.

This recipe takes a different approach for **vision and embedding**:

| Component | Qwen | Ministral | Why |
|-----------|------|-----------|-----|
| Text decoder | Olive/ModelBuilder (`text.json`) | Olive/ModelBuilder (`text.json`) | Same — ModelBuilder handles GQA + quantization |
| Vision encoder | Olive: PyTorch export + 5-6 passes | **Olive/MobiusBuilder** (`vision_embedding_export.json`) + Olive INT8 (`vision.json`) | Pixtral's dynamic image dims break `torch.onnx.export` |
| Embedding | Olive: PyTorch export + 5 passes | **Olive/MobiusBuilder** export (FP16/FP32, no quantization) | Olive's GatherBlockQuantized has data format bugs |

**Why does Ministral use MobiusBuilder instead of standard Olive export?** The Olive
`MobiusBuilder` pass constructs the ONNX graph declaratively (via the
[mobius](https://github.com/onnxruntime/mobius) library internally) rather than
tracing through PyTorch. The resulting models already contain the graph optimizations
that Qwen's Olive passes spend 5-6 steps creating:

- **Fused operators:** `MultiHeadAttention`, `SkipSimplifiedLayerNormalization`,
  `RotaryEmbedding` — already present in MobiusBuilder output (Qwen achieves these via
  `OrtTransformersOptimization`)
- **FP16 weights:** all 840M vision params exported as FP16 directly (Qwen
  converts from FP32 via `OnnxFloatToFloat16`)
- **Clean graph:** 0 Gemm nodes, 0 redundant Cast chains (Qwen cleans these
  via `GemmToMatMulAdd` and `OnnxPeepholeOptimizer`)
- **No PyTorch export artifacts:** no `PackedAttentionToLoopMHA` surgery needed
  since MobiusBuilder doesn't go through dynamo

**What Olive still handles:** `vision.json` applies
`OnnxBlockWiseRtnQuantization` (INT8) to the MobiusBuilder-exported FP16 vision model
for all targets (cuda, webgpu, cpu_and_mobile).

**Why optimize.py has more lines (~400) than Qwen (~170):**

| Code section | Lines | Why it can't be JSON-driven |
|---|---|---|
| `export_vision_and_embedding()` | ~55 | Runs Olive/MobiusBuilder then reorganizes flat output into subdirectory layout expected by quantization pass |
| `update_genai_config()` | ~150 | Olive generates decoder config only; VLM 3-model config + transforms-based processor_config has no Olive pass |
| `quantize_vision_and_embedding()` | ~25 | Post-export INT8 on pre-built ONNX (Olive JSON-driven, but needs orchestration + cleanup) |
| `fix_tokenizer()` | ~15 | No Olive tokenizer patching pass |

The text decoder export (`text.json`) and INT8 quantization (`vision.json`) ARE Olive JSON-driven — identical to Qwen.

## Known Limitations

- **CPU vision: language drift on some images.** The quantized vision encoder occasionally produces embeddings that cause the text decoder to respond in the wrong language (e.g., Chinese instead of English). This has been observed on specific test images and is a known artifact of vision quantization. INT8 significantly reduces this compared to INT4.
- **CUDA vision quantization is parameter-sensitive.** Symmetric block-128 INT8 vision caused a large quality drop (56.40% on 500 AI2D samples). The CUDA recipe uses asymmetric block-32 INT8 vision, which recovered the result to 73.00% and closely tracks the unquantized Mobius FP16 vision package.
- **FP8 checkpoint requires special kernels.** This recipe defaults to the `-BF16` checkpoint. The FP8 checkpoint can require CUDA kernels that are not available on all machines.

## Notes

- **Multi-image supported.** The runtime supports variable-count multi-image inputs via PixtralImageSizes metadata. Requires onnxruntime-extensions ≥ PR #1050 and models exported with PixtralImageSizes in `processor_config.json`.

- **CPU pipeline**: MobiusBuilder exports FP16 as an intermediate format. Olive then quantizes vision to INT8. For CPU deployment, the cpu_and_mobile JSON configs set `precision: fp32` so embedding outputs float32 natively (CPU EP promotes FP16 to FP32, which causes genai dtype mismatches). The `--dtype` flag is accepted for backward compatibility but does not control export precision — precision is set in the JSON config files.
- **CUDA/WebGPU pipeline**: MobiusBuilder exports FP16 directly for vision/embedding. Olive quantizes vision to asymmetric block-32 INT8. Text decoder uses k_quant_mixed INT4 via ModelBuilder.
- The FP8 Hugging Face checkpoint uses quantized weights. Use the default `-BF16` checkpoint unless you specifically need to test FP8 export behavior.
- The tokenizer uses `TokenizersBackend` class which genai doesn't support. The optimize script fixes this to `LlamaTokenizer`.
- Pixtral vision supports dynamic image sizes (multiples of 28, up to 1540×1540).
- The text decoder includes `llama_4_attn_scale` for long-context attention (>16K tokens).
