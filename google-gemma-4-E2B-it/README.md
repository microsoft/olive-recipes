# Gemma 4 E2B (google/gemma-4-E2B-it)

Olive recipes that export [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
to ONNX via the [`MobiusBuilder`](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
pass and (optionally) quantize the decoder with Olive's K-Quant (Q4_K_M)
pass for INT4 deployment.

Gemma 4 is an any-to-any multimodal model with vision, audio, and text
capabilities. The pipeline produces four ONNX components (decoder,
vision_encoder, audio_encoder, embedding) for use with ORT GenAI.
`MobiusBuilder` writes a fully-formed ORT GenAI package
(`genai_config.json`, `tokenizer.json`, `image_processor.json`,
`audio_feature_extraction.json`) alongside the ONNX files — no
post-processing required.

## Prerequisites

```bash
pip install olive-ai mobius-ai
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| CPU | `pip install onnxruntime-genai` |
| GPU (CUDA) | `pip install onnxruntime-genai-cuda` |

## Recipes

| Recipe | Pipeline | Output dir |
|---|---|---|
| `cpu/fp32/config.json` | `MobiusBuilder(fp32)` | `cpu/fp32/models` |
| `cpu/int4/config.json` | `MobiusBuilder(fp32)` → `OnnxKQuantQuantization(bits=4, block=32)` | `cpu/int4/models` |
| `cuda/fp16/config.json` | `MobiusBuilder(fp16)` | `cuda/fp16/models` |
| `cuda/int4/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block=32)` | `cuda/int4/models` |

### Mixed quantization (separate text / vision / audio)

These recipes split the model into components with per-component
quantization — int4 for the text decoder, int8 for vision and audio
encoders — for better accuracy vs. latency trade-offs.

| Recipe | Pipeline | Output dir |
|---|---|---|
| `cpu/mixed/export.json` | `MobiusBuilder(fp32)` — export all components | `cpu/mixed/models` |
| `cpu/mixed/text.json` | `OnnxKQuantQuantization(int4, block=32)` — quantize decoder | `cpu/mixed/models/decoder` |
| `cpu/mixed/vision.json` | `OnnxBlockWiseRtnQuantization(int8, block=128)` — quantize vision encoder | `cpu/mixed/models/vision_encoder` |
| `cpu/mixed/audio.json` | `OnnxBlockWiseRtnQuantization(int8, block=128)` — quantize audio encoder | `cpu/mixed/models/audio_encoder` |
| `cuda/mixed/export.json` | `MobiusBuilder(fp16)` — export all components | `cuda/mixed/models` |
| `cuda/mixed/text.json` | `OnnxKQuantQuantization(int4, block=32)` — quantize decoder | `cuda/mixed/models/decoder` |
| `cuda/mixed/vision.json` | `OnnxBlockWiseRtnQuantization(int8, block=32)` — quantize vision encoder | `cuda/mixed/models/vision_encoder` |
| `cuda/mixed/audio.json` | `OnnxBlockWiseRtnQuantization(int8, block=32)` — quantize audio encoder | `cuda/mixed/models/audio_encoder` |

**Run order**: export first, then text, vision, and audio (the latter three
can run in parallel):

```bash
# CPU mixed
olive run --config cpu/mixed/export.json
olive run --config cpu/mixed/text.json
olive run --config cpu/mixed/vision.json
olive run --config cpu/mixed/audio.json

# CUDA mixed
olive run --config cuda/mixed/export.json
olive run --config cuda/mixed/text.json
olive run --config cuda/mixed/vision.json
olive run --config cuda/mixed/audio.json
```

K-Quant (Q4_K_M) is significantly faster with GPU acceleration —
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
├── vision_encoder/model.onnx   # Vision encoder
├── audio_encoder/model.onnx    # Audio encoder
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

# CUDA mixed (int4 decoder + int8 vision/audio)
python inference.py --device gpu --variant mixed --prompt "Explain quantum computing"

# Interactive mode
python inference.py --device gpu --variant mixed --interactive
```

## Evaluation

### MMLU (text)

Run through Olive with the `LMEvaluator` configs in `eval/` (mixed model):

```bash
olive run --config eval/mmlu_cpu.json    # CPU
olive run --config eval/mmlu_cuda.json   # CUDA
```

Or use the standalone script (also supports `--task`, `--limit`, and other variants):

```bash
# MMLU (5-shot, default 100 samples), CPU
python eval.py

# CUDA INT4
python eval.py --device gpu --variant int4

# CUDA mixed
python eval.py --device gpu --variant mixed
```

### Vision — AI2D (exact_match)

> **Note**: The `olive run` eval configs require Olive to support nested model
> layouts in the evaluator's genai_config.json discovery. Until then, use
> a custom evaluation script.

### Audio — FLEURS ASR (WER)

> **Note**: The `olive run` audio eval configs require Olive to add gemma4
> as a supported model type in the speech evaluator. Until then, use a custom
> evaluation script.
```

## References

- Mobius docs: <https://github.com/onnxruntime/mobius>
- Olive `MobiusBuilder` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py>
- Olive `OnnxKQuantQuantization` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/kquant_quantization.py>
