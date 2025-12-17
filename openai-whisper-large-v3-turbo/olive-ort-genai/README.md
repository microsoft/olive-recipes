# Whisper Large V3 Turbo Optimization

This example demonstrates how to export and optimize OpenAI's Whisper Large V3 Turbo model to ONNX format using Olive, with FP16 quantization.

## Model Information

- **Model**: `openai/whisper-large-v3-turbo`
- **Architecture**: 4 decoder layers (compared to 32 in whisper-large-v3)
- **Optimizations**: FP16 precision, external data format for efficient storage

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Export Models

```bash
python whisper.py
```

This will create ONNX models in the `models/` directory:
- `whisper-large-v3-turbo_encoder_fp16.onnx` - Encoder model with cross-attention KV cache initialization
- `whisper-large-v3-turbo_decoder_fp16.onnx` - Decoder with past KV cache (for autoregressive generation)

### 2. Run Inference

```bash
python test_transcription.py
```

Expected output:
```
Transcription: the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about
```

## Key Features

- **External Data Format**: Large model weights stored separately in `.onnx.data` files
- **Optimized KV Cache**: Encoder outputs cross-attention KV caches that are reused across decoder steps
- **Dynamic Shapes**: Support for variable batch sizes and sequence lengths
- **Post-processing**: Automatic fixing of dimension parameters and missing inputs
