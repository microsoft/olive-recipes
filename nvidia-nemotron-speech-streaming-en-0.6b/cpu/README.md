# Nemotron Speech Streaming (CPU EP, INT4)

This recipe exports **nvidia/nemotron-speech-streaming-en-0.6b** to ONNX, optimizes the encoder, and produces CPU-ready artifacts.

## Files
- `cpu/nemotron_speech_int4_cpu_kquant.json` – recipe definition
- `scripts/export_nemotron_to_onnx_static_shape.py` – ONNX export (streaming/static-shape)
- `scripts/optimize_encoder.py` – encoder graph fusion + dtype conversion/quantization
- `scripts/test_e2e.py` – e2e smoke test

## Setup
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nvidia-nemotron-speech-streaming-en-0.6b/cpu/requirements.txt
```

## Run manually

```bash
cd nvidia-nemotron-speech-streaming-en-0.6b

python scripts/export_nemotron_to_onnx_static_shape.py \
  --model_name nvidia/nemotron-speech-streaming-en-0.6b \
  --output_dir build/onnx_models_fp32 \
  --streaming \
  --chunk_size 0.56 \
  --left_chunks 10 \
  --device cpu

python scripts/optimize_encoder.py \
  --model_dir build/onnx_models_fp32 \
  --output_dir build/onnx_models_int4 \
  --dtype int4 \
  --quant_method k_quant_mixed \
  --block_size 32 \
  --accuracy_level 4

python scripts/test_e2e.py --model_dir build/onnx_models_int4
```

## Output
Expected optimized artifacts in:
- `build/onnx_models_int4/encoder.onnx`
- `build/onnx_models_int4/decoder.onnx`
- `build/onnx_models_int4/joint.onnx`
- `build/onnx_models_int4/genai_config.json`
- `build/onnx_models_int4/audio_processor_config.json`
