# Nemotron Speech Streaming (CPU EP, INT4)

This recipe exports **nvidia/nemotron-speech-streaming-en-0.6b** to ONNX and
optimizes the encoder for low-memory CPU inference using Olive.

The pipeline produces three sub-models (encoder, decoder, joint), applies
Conformer-specific graph fusions and INT4 block-wise RTN quantization to the
encoder, and leaves the decoder and joint networks in FP32.

## Prerequisites

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI for CPU inference:

```bash
pip install onnxruntime-genai --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple
```

> **Note:** NeMo requires Cython and packaging to be installed before
> `nemo_toolkit[asr]`. Install them first if you encounter build errors:
> `pip install Cython packaging`

## Steps

### 1. Export & Optimize

`optimize.py` orchestrates the two-stage pipeline.

| Command | Description |
|---------|-------------|
| `python optimize.py` | Full pipeline: export from NeMo + Olive INT4 optimize |
| `python optimize.py --skip-export` | Olive optimization only (models already exported) |
| `python optimize.py --chunk-size 1.12 --left-chunks 10` | Custom streaming chunk size |

Run from this directory (`cpu/`):

```bash
cd nvidia-nemotron-speech-streaming-en-0.6b/cpu
python optimize.py
```

#### What `optimize.py` does

**Stage 1 — Export (NeMo → ONNX):**
Runs `scripts/export_nemotron_to_onnx_static_shape.py` to export the encoder
(with static-shape streaming cache I/O), decoder (stateful LSTM), joint network,
tokenizer, and `genai_config.json` / `audio_processor_config.json` to
`build/onnx_models_fp32/`.

**Stage 2 — Olive Optimization (ONNX → INT4 ONNX):**
Runs the Olive passes defined in `encoder.json`:
- **OrtTransformersOptimization** (`model_type="conformer"`): Fuses Conformer
  attention subgraphs into `MultiHeadAttention`, `SkipLayerNormalization`,
  `BiasGelu`, etc.
- **OnnxBlockWiseRtnQuantization**: INT4 weight quantization
  (`block_size=32`, symmetric, `accuracy_level=4`).

The decoder and joint networks are copied unchanged to `build/onnx_models_int4/`.

### 2. Run Inference

Use ONNX Runtime GenAI with the optimized model directory:

```bash
# From the nvidia-nemotron-speech-streaming-en-0.6b/ directory
python scripts/test_e2e.py --model_dir cpu/build/onnx_models_int4
```

## Directory Structure

```
nvidia-nemotron-speech-streaming-en-0.6b/
├── LICENSE
├── cpu/
│   ├── encoder.json         # Olive config: graph fusion + INT4 quantization
│   ├── info.yml             # Recipe metadata
│   ├── optimize.py          # Main script: export + Olive pipeline
│   ├── requirements.txt
│   └── README.md
└── scripts/
    ├── export_nemotron_to_onnx_static_shape.py  # NeMo → ONNX export
    ├── optimize_encoder.py                       # Standalone encoder optimizer
    ├── test_e2e.py                               # E2E smoke test (onnxruntime-genai)
    ├── test_real_speech.py                       # Real-audio test
    └── README.md
```

## Output

Optimized artifacts written to `build/onnx_models_int4/`:

| File | Description |
|------|-------------|
| `encoder.onnx` + `.data` | FastConformer encoder (INT4, Olive-optimized) |
| `decoder.onnx` | RNNT prediction network (FP32, stateful LSTM h/c I/O) |
| `joint.onnx` | Joint network (FP32) |
| `genai_config.json` | Model configuration for onnxruntime-genai |
| `audio_processor_config.json` | Mel spectrogram parameters (16 kHz, 128 mels) |
| `tokenizer.json` / `tokenizer_config.json` | HuggingFace Unigram tokenizer |
