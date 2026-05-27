# Nemotron Scripts

Utility scripts for the Nemotron 3.5 ASR Streaming Multilingual 0.6B recipe.

All ONNX export is now handled through Olive configs — see `src/README.md`
for the full pipeline.

## Prerequisites

```bash
conda create -n nemotron-export python=3.10 -y
conda activate nemotron-export
pip install Cython packaging torch torchaudio onnxruntime
pip install "nemo_toolkit[asr]>=2.7.1"
```

## Export (via Olive)

From the `nvidia-nemotron-asr-streaming-multilingual-0.6b` directory:

```bash
python src/optimize.py
```

This exports all components (encoder, decoder, joint, tokenizer, configs)
through Olive's declarative pass system. See `src/README.md` for details.

## Output Files

| File | Description |
|------|-------------|
| `silero_vad.onnx` | Silero VAD model (downloaded from onnx-community/silero-vad) |
| `encoder.onnx` (+`.data`) | Multilingual streaming Conformer encoder (INT4 k-quant by default) |
| `decoder.onnx` (+`.data`) | RNNT prediction network (stateful LSTM h/c I/O, FP32) |
| `joint.onnx` (+`.data`) | Joint network (encoder + decoder → logits, FP32) |
| `genai_config.json` | Model configuration for onnxruntime-genai (includes per-language prompt IDs) |
| `audio_processor_config.json` | Mel spectrogram parameters (16 kHz, 128 mels, 512 FFT) |
| `model_config.json` | Architecture metadata used by genai |
| `tokenizer.json` | HuggingFace Unigram tokenizer (multilingual vocab) |
| `tokenizer_config.json` | T5Tokenizer class routing for ORT Extensions |
| `vocab.txt` | Raw vocabulary (one token per line) |

## Scripts

| Script | Purpose |
|--------|---------|
| `export_tokenizer.py` | Extract vocab from NeMo and create ORT-compatible tokenizer |
