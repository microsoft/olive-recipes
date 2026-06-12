# Gemma 4 12B Unified (google/gemma-4-12B-it)

Olive recipes that export [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)
to ONNX via the [`MobiusBuilder`](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
pass and (optionally) quantize the decoder with Olive's K-Quant (Q4_K_M)
pass for INT4 deployment.

Gemma 4 12B is the **Unified** member of the Gemma 4 family: an
**encoder-free**, any-to-any multimodal model with text, image, and audio
input. Unlike the E2B/E4B variants — which use dedicated vision/audio
encoders — the 12B Unified model projects raw image patches and audio
waveform features directly into the decoder's embedding space through
lightweight linear layers. It supports a context window of up to 256K
tokens.

The pipeline still produces four ONNX components for use with ORT GenAI:

- `decoder` — the decoder-only transformer.
- `embedding` — text embedding + multimodal fusion.
- `vision_encoder` — encoder-free image embedder: raw merged pixel patches
  (`pixel_values`) + integer patch coordinates (`pixel_position_ids`) →
  `image_features`.
- `audio_encoder` — encoder-free audio embedder: raw waveform-frame
  features (`input_features`) + a validity mask (`input_features_mask`) →
  `audio_features`.

`MobiusBuilder` writes a fully-formed ORT GenAI package
(`genai_config.json`, `tokenizer.json`, `image_processor.json`,
`audio_feature_extraction.json`) alongside the ONNX files — no
post-processing required.

## Prerequisites

```bash
pip install -r requirements.txt
```

`requirements.txt` pulls in `olive-ai[gpu]`, `mobius-ai`, and `lm-eval`.

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| CPU | `pip install onnxruntime-genai` |
| GPU (CUDA) | `pip install onnxruntime-genai-cuda` |

## Recipes

| Recipe | Pipeline | Output dir |
|---|---|---|
| `cpu/fp32/config.json` | `MobiusBuilder(fp32)` | `cpu/fp32/models` |
| `cpu/int4/config.json` | `MobiusBuilder(fp32)` → `OnnxKQuantQuantization(bits=4, block_size=32)` | `cpu/int4/models` |
| `cuda/fp16/config.json` | `MobiusBuilder(fp16)` | `cuda/fp16/models` |
| `cuda/int4/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block_size=32)` | `cuda/int4/models` |

At 12B parameters this model is sized for consumer GPUs and workstations.
INT4 (K-Quant) is recommended for most deployments; the FP16 decoder alone
is ~24 GB. K-Quant (Q4_K_M) is significantly faster with GPU acceleration —
install `cupy-cuda12x` for a 19–51× speedup during quantization.

## Build

```bash
# CPU, full precision
olive run --config cpu/fp32/config.json

# CPU, INT4 (K-Quant)
olive run --config cpu/int4/config.json

# CUDA, FP16
olive run --config cuda/fp16/config.json

# CUDA, INT4 (K-Quant)
olive run --config cuda/int4/config.json
```

Each command produces the full ORT GenAI package in the recipe's
`output_dir`:

```
<output_dir>/
├── decoder/model.onnx          # Text decoder
├── vision_encoder/model.onnx   # Encoder-free image embedder
├── audio_encoder/model.onnx    # Encoder-free audio embedder
├── embedding/model.onnx        # Embedding fusion
├── genai_config.json           # Runtime configuration
├── image_processor.json
├── audio_feature_extraction.json
├── tokenizer.json
└── tokenizer_config.json
```

## Inference

```bash
# Text-only (CPU, fp32)
python inference.py --prompt "What is the capital of France?"

# CPU INT4
python inference.py --variant int4 --prompt "Hello"

# CUDA INT4
python inference.py --device gpu --variant int4 --prompt "Explain quantum computing"

# Interactive mode
python inference.py --device gpu --variant int4 --interactive
```

## Evaluation

```bash
# MMLU Pro (default 100 samples), CPU
python eval.py

# CUDA INT4
python eval.py --device gpu --variant int4
```

Published reference scores for the 12B Unified model (Google-reported, CoT):

| Benchmark | Score |
|---|---|
| MMLU Pro | 77.2% |
| GPQA Diamond | 78.8% |

## References

- Mobius docs: <https://github.com/onnxruntime/mobius>
- Olive `MobiusBuilder` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py>
- Olive `OnnxKQuantQuantization` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/kquant_quantization.py>
