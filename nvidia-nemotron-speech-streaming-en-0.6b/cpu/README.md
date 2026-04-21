# Nemotron Speech Streaming (CPU EP, INT4)

This recipe exports **nvidia/nemotron-speech-streaming-en-0.6b** to ONNX, optimizes the encoder, and produces CPU-ready artifacts.

## License

This model has an NVIDIA Open Model License Agreement. The contents of the license agreement can be found [here](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Files
- `cpu/nemotron_speech_int4_cpu.json` – Olive workflow config (OnnxConversion → graph fusion → INT4 quantization)
- `cpu/nemotron_encoder_load.py` – model loader script for Olive (loads encoder with streaming wrapper)
- `cpu/optimize.py` – full pipeline script (export decoder/joint/tokenizer + Olive encoder + assembly)
- `scripts/export_nemotron_to_onnx_static_shape.py` – ONNX export (streaming/static-shape)
- `scripts/export_tokenizer.py` – tokenizer export
- `scripts/test_e2e.py` – e2e smoke test

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
1. **Export** — NeMo model → ONNX (encoder, decoder, joint, tokenizer, configs)
2. **Optimize encoder** — Olive converts, fuses, and INT4-quantizes the encoder
   using built-in passes (OnnxConversion → OrtTransformersOptimization →
   OnnxKQuantQuantization) via `cpu/nemotron_speech_int4_cpu.json`
3. **Assemble** — copies decoder, joint, tokenizer, configs, and Silero VAD
   into the final output directory

To skip the NeMo export step if models are already in `cpu/build/onnx_models_fp32/`:

```bash
python cpu/optimize.py --skip-export
```

## Output
Expected optimized artifacts in:
- `cpu/build/onnx_models_int4/encoder.onnx`
- `cpu/build/onnx_models_int4/decoder.onnx`
- `cpu/build/onnx_models_int4/joint.onnx`
- `cpu/build/onnx_models_int4/silero_vad.onnx`
- `cpu/build/onnx_models_int4/genai_config.json`
- `cpu/build/onnx_models_int4/audio_processor_config.json`
- `cpu/build/onnx_models_int4/tokenizer.json`
- `cpu/build/onnx_models_int4/tokenizer_config.json`
- `cpu/build/onnx_models_int4/vocab.txt`
