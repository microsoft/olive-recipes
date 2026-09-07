# Nemotron Scripts

Test and utility scripts for the Nemotron Speech Streaming recipe.

All ONNX export is now handled through Olive configs — see `cpu/README.md`
for the full pipeline.

## Prerequisites

```bash
conda create -n nemotron-export python=3.10 -y
conda activate nemotron-export
pip install Cython packaging torch torchaudio onnxruntime
pip install "nemo_toolkit[asr]>=2.7.1"
```

## Export (via Olive)

From the `nvidia-nemotron-speech-streaming-en-0.6b` directory:

```bash
python cpu/optimize.py
```

This exports all components (encoder, decoder, joint, tokenizer, configs)
through Olive's declarative pass system. See `cpu/README.md` for details.

## Test

```bash
# End-to-end test via onnxruntime-genai (requires built wheel)
python scripts/test_e2e.py

# Real speech test with jfk.flac
python scripts/test_real_speech.py
```

## Output Files

| File | Description |
|------|-------------|
| `silero_vad.onnx` | Silero VAD model (downloaded from onnx-community/silero-vad) |
| `encoder.onnx` (+`.data`) | FastConformer encoder (24 layers, INT4 quantized) |
| `decoder.onnx` (+`.data`) | RNNT prediction network (2 LSTM layers, stateful h/c I/O, FP32) |
| `joint.onnx` (+`.data`) | Joint network (encoder + decoder → logits, FP32) |
| `genai_config.json` | Model configuration for onnxruntime-genai |
| `audio_processor_config.json` | Mel spectrogram parameters (16kHz, 128 mels, 512 FFT) |
| `tokenizer.json` | HuggingFace Unigram tokenizer (1025 tokens) |
| `tokenizer_config.json` | T5Tokenizer class routing for ORT Extensions |
| `vocab.txt` | Raw vocabulary (one token per line) |

## Scripts

| Script | Purpose |
|--------|---------|
| `export_tokenizer.py` | Extract vocab from NeMo and create ORT-compatible tokenizer |
| `test_e2e.py` | End-to-end test: model load, tokenizer, inference, raw ONNX baseline |
| `test_real_speech.py` | Real speech test with NeMo preprocessing, compares OG vs raw ORT |
