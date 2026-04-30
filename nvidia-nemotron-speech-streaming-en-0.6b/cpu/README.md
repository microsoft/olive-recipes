# Nemotron Speech Streaming (CPU EP, INT4)

This recipe exports **nvidia/nemotron-speech-streaming-en-0.6b** to ONNX, optimizes the encoder, and produces CPU-ready artifacts.

All model components are handled through Olive's declarative pass system:
- **Encoder**: OnnxConversion → OrtTransformersOptimization (Conformer fusion) → OnnxKQuantQuantization (INT4)
- **Decoder**: OnnxConversion (FP32)
- **Joint**: OnnxConversion (FP32)

## Files
- `cpu/nemotron_speech_int4_cpu.json` – Olive encoder config (convert → fusion → INT4)
- `cpu/nemotron_decoder_fp32_cpu.json` – Olive decoder config (convert only)
- `cpu/nemotron_joint_fp32_cpu.json` – Olive joint config (convert only)
- `cpu/nemotron_model_load.py` – model loaders + dummy inputs for all components
- `cpu/optimize.py` – full pipeline script (Olive × 3 + tokenizer + configs + VAD)
- `scripts/export_tokenizer.py` – tokenizer export
- `scripts/test_e2e.py` – e2e smoke test
- `scripts/test_real_speech.py` – real speech test

## Setup
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nvidia-nemotron-speech-streaming-en-0.6b/cpu/requirements.txt
```

## Run

From the `nvidia-nemotron-speech-streaming-en-0.6b` directory:

```bash
cd nvidia-nemotron-speech-streaming-en-0.6b

python cpu/optimize.py
```

This runs the full pipeline:
1. **Encoder** — Olive: OnnxConversion → graph fusion → INT4 quantization
2. **Decoder** — Olive: OnnxConversion (FP32)
3. **Joint** — Olive: OnnxConversion (FP32)
4. **Tokenizer** — exports vocab + tokenizer.json
5. **Configs** — generates genai_config.json + audio_processor_config.json
6. **VAD** — downloads Silero VAD ONNX model

Or run individual components directly with Olive CLI:

```bash
python -m olive run --config cpu/nemotron_speech_int4_cpu.json
python -m olive run --config cpu/nemotron_decoder_fp32_cpu.json
python -m olive run --config cpu/nemotron_joint_fp32_cpu.json
```

## Output
Expected optimized artifacts in `cpu/build/onnx_models_int4/`:
- `encoder.onnx` (INT4 k-quant)
- `decoder.onnx` (FP32)
- `joint.onnx` (FP32)
- `silero_vad.onnx`
- `genai_config.json`
- `audio_processor_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`
