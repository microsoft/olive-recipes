# TranslateGemma-4B-IT ONNX Recipe

Export and run [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) as a full vision-language model (VLM) with ONNX Runtime GenAI.

TranslateGemma is a translation model supporting 55 languages, with both **text-to-text** and **image-to-text** translation capabilities. This recipe exports it as three separate ONNX sub-models (text decoder, vision encoder, embedding) that run together through the ORT GenAI multimodal pipeline.

## Prerequisites

```bash
pip install olive-ai onnxruntime-genai transformers torch
```

## Quick Start

### 1. Authenticate with Hugging Face

TranslateGemma is a [gated model](https://huggingface.co/google/translategemma-4b-it). Accept the license on Hugging Face, then log in so models download automatically during export:

```bash
huggingface-cli login
```

### 2. Export to ONNX

From the `builtin/` directory:

```bash
cd builtin

# INT4 RTN text decoder + FP32 vision/embedding (recommended, ~6.7 GB total)
python optimize.py --config-dir cpu_and_mobile

# AWQ INT4 text decoder + FP32 vision/embedding (~6.4 GB total)
python optimize.py --config-dir cpu_and_mobile_awq

# Full FP32 baseline (~19.2 GB total)
python optimize.py --config-dir cpu_and_mobile_fp32
```

### 3. Run inference

```bash
# Text translation (default: cpu_and_mobile)
python inference.py --source-lang en --target-lang es --text "Hello, how are you?"

# Image translation
python inference.py --source-lang en --target-lang fr --image <image-path>

# Use AWQ model
python inference.py --model-dir builtin/cpu_and_mobile_awq/models --source-lang en --target-lang de --text "Good morning"

# Use FP32 model
python inference.py --model-dir builtin/cpu_and_mobile_fp32/models --source-lang en --target-lang ja --text "Good morning"
```

## Export Configurations

Three export configurations are provided, all producing the same three-ONNX-model VLM layout:

| Config | Text Decoder | Embedding | Vision | Total Size |
|---|---|---|---|---|
| `cpu_and_mobile` | INT4 RTN (block 128) | FP32 | FP32 | ~6.7 GB |
| `cpu_and_mobile_awq` | AWQ INT4 (block 128) | FP32 | FP32 | ~6.4 GB |
| `cpu_and_mobile_fp32` | FP32 | FP32 | FP32 | ~19.2 GB |

Each produces a `models/` directory containing:

```
models/
  text.onnx              # Text decoder (34 Gemma3 layers + LM head)
  text.onnx.data         # External weights
  embedding.onnx         # Token embedding + image feature scattering
  embedding.onnx.data
  vision.onnx            # SigLIP vision encoder + multimodal projector
  vision.onnx.data
  genai_config.json      # Runtime config for ORT GenAI
  processor_config.json  # Image preprocessing pipeline
  tokenizer.json         # Tokenizer files
  tokenizer_config.json
```

## Architecture

TranslateGemma is a `Gemma3ForConditionalGeneration` multimodal model with three components:

```
Image [B, 3, 896, 896]
  |
  v
vision.onnx (SigLIP 27 layers + AvgPool2d projector)
  |
  v  image_features [B*256, 2560]
  |
  +--- input_ids [B, seq_len] --->  embedding.onnx (embed_tokens + scatter)
                                      |
                                      v  inputs_embeds [B, seq_len, 2560]
                                      |
                                      +---> text.onnx (34 Gemma3 decoder layers)
                                              |
                                              v  logits -> tokens -> translation
```

- **Vision**: SigLIP encoder (27 layers, 1152-dim) processes 896x896 images into 4096 patches, then a projector (AvgPool2d + RMSNorm + linear) compresses to 256 tokens at 2560-dim.
- **Embedding**: Looks up token embeddings (scaled by sqrt(2560)), then scatters vision features into image-token positions.
- **Text**: Standard Gemma3 decoder with 34 layers, sliding/full attention pattern, generating translation tokens autoregressively.

## Supported Languages

TranslateGemma supports translation across 55 languages including: Arabic, Bengali, Bulgarian, Catalan, Chinese (Simplified/Traditional), Czech, Danish, Dutch, English, Estonian, Farsi, Filipino, Finnish, French, German, Greek, Gujarati, Hebrew, Hindi, Croatian, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Korean, Latvian, Lithuanian, Malayalam, Marathi, Norwegian, Pashto, Polish, Portuguese (BR/PT), Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Zulu.

## Benchmarking

Evaluate translation quality against WMT24++ using COMET:

```bash
pip install unbabel-comet datasets

# Quick check: 3 language pairs, 50 segments each (~10 min CPU)
python benchmark_wmt24pp.py --lang-pairs en-de_DE en-es_MX en-fr_FR --max-segments 50

# Full benchmark: all 55 pairs, stratified 150 segments each (~6h CPU)
python benchmark_wmt24pp.py --lang-pairs all --max-segments 150 --seed 42

# Compare AWQ vs RTN
python benchmark_wmt24pp.py --model-dir builtin/cpu_and_mobile_awq/models --output awq_results.json
```

Model card reported scores (4B, WMT24++ 55 langs):
- MetricX: 5.32 (lower is better)
- COMET: 81.6 (higher is better)

## File Structure

```
translategemma-4b-it/
  data/                           # Test images
  builtin/
    optimize.py                   # Export orchestration (3 Olive pipelines + config assembly)
    user_script.py                # Vision/embedding wrapper modules for Olive export
    cpu_and_mobile/               # INT4 RTN configs (text: google/translategemma-4b-it)
    cpu_and_mobile_awq/           # AWQ configs (text: amd/TranslateGemma-4b-it-awq-quant-uint4-wo-128)
    cpu_and_mobile_fp32/          # FP32 configs (text: google/translategemma-4b-it)
  inference.py                    # Text and image translation inference
  benchmark_wmt24pp.py            # WMT24++ COMET evaluation
  README.md
```

Models are downloaded automatically from Hugging Face during export. Ensure you have accepted the [Gemma license](https://huggingface.co/google/translategemma-4b-it) and are logged in via `huggingface-cli login`.

## How It Works

The export pipeline (`optimize.py`) runs three Olive workflows sequentially:

1. **Text decoder** via `ModelBuilder` pass -- reads the PyTorch model and constructs an optimized ONNX graph with KV-cache support. For INT4 variants, weights are quantized during this step.
2. **Embedding model** via `OnnxConversion` -- exports a custom `nn.Module` wrapper that combines the token embedding layer with image-feature scattering logic.
3. **Vision model** via `OnnxConversion` -- exports the SigLIP vision tower and multimodal projector as a single ONNX graph.

After export, `optimize.py` patches `genai_config.json` to register all three sub-models and creates `processor_config.json` for the C++ image preprocessing pipeline (resize 896x896, normalize to [-1,1], HWC-to-CHW permute).
