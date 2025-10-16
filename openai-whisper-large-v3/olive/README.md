# Whisper Large V3 Optimization

This example demonstrates how to export and optimize OpenAI's Whisper Large V3 model to ONNX format using Olive, with INT8 quantization.

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Export Models (Without Decoder Merging)

```bash
python whisper.py
```

This will create three ONNX models in the `models/` directory:
- `encoder/model.onnx` - Encoder model (INT8 quantized)
- `decoder/model.onnx` - Decoder without past KV cache (INT8 quantized)
- `decoder_with_past/model.onnx` - Decoder with past KV cache (INT8 quantized)

### 2. Run Inference with Separate Decoders

```bash
python test_transcription.py
```

Expected output:
```
Transcription: the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about
```

### 3. Export Models with Merged Decoder (Optional)

```bash
python whisper.py --merge_decoder
```

This performs additional post-processing:
1. Post-processes decoder models to ensure correct output structure
2. Fixes dynamic axes for proper shape inference
3. Merges `decoder` and `decoder_with_past` into `decoder_merged/model.onnx`

### 4. Run Inference with Merged Decoder

```bash
python test_transcription.py --merge_decoder
```
