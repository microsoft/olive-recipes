# Nemotron 3.5 ASR Streaming Multilingual 0.6B (INT4, CPU/CUDA)

This recipe exports **nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b**
(100+ languages) to ONNX, optimizes the encoder, and produces deployment-ready
artifacts for `onnxruntime-genai`.

All model components are handled through Olive's declarative pass system:
- **Encoder**: OnnxConversion → OnnxKQuantQuantization (INT4 default, or INT8 dynamic)
- **Decoder**: OnnxConversion (FP32)
- **Joint**: OnnxConversion (FP32)

## Files
- `src/nemotron_encoder_int4_cpu.json` – Olive encoder config (convert → INT4 k-quant)
- `src/nemotron_encoder_int8_cpu.json` – Olive encoder config (convert → INT8 dynamic, optional)
- `src/nemotron_decoder_fp32_cpu.json` – Olive decoder config (convert only)
- `src/nemotron_joint_fp32_cpu.json` – Olive joint config (convert only)
- `src/nemotron_model_load.py` – model loaders + dummy inputs for all components
- `src/optimize.py` – full pipeline script (Olive × 3 + tokenizer + configs + VAD)
- `scripts/export_tokenizer.py` – tokenizer export

## Setup
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nvidia-nemotron-asr-streaming-multilingual-0.6b/src/requirements.txt
```

## Run

From the `nvidia-nemotron-asr-streaming-multilingual-0.6b` directory:

```bash
cd nvidia-nemotron-asr-streaming-multilingual-0.6b

# Full pipeline, INT4 encoder (default)
python src/optimize.py

# Or INT8 encoder
python src/optimize.py --encoder-precision int8

# Custom output directory
python src/optimize.py --output-dir build/multilingual_onnx_int4
```

This runs the full pipeline:
1. **Encoder** — Olive: OnnxConversion → INT4/INT8 quantization
2. **Decoder** — Olive: OnnxConversion (FP32)
3. **Joint** — Olive: OnnxConversion (FP32)
4. **Tokenizer** — exports vocab + tokenizer.json
5. **Configs** — generates genai_config.json + audio_processor_config.json
6. **VAD** — downloads Silero VAD ONNX model

Or run individual components directly with Olive CLI:

```bash
python -m olive run --config src/nemotron_encoder_int4_cpu.json
python -m olive run --config src/nemotron_encoder_int8_cpu.json
python -m olive run --config src/nemotron_decoder_fp32_cpu.json
python -m olive run --config src/nemotron_joint_fp32_cpu.json
```

## Output
Expected optimized artifacts in `src/build/onnx_models_int4/` (default output directory):
- `encoder.onnx` (INT4 k-quant, ~660 MB)
- `decoder.onnx` (FP32, ~57 MB)
- `joint.onnx` (FP32, ~36 MB)
- `silero_vad.onnx` (~2 MB)
- `genai_config.json`
- `audio_processor_config.json`
- `model_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`

Total size: ~760 MB (INT4 encoder).

## Inference

Use the multilingual-aware example from `onnxruntime-genai`, passing a `--language` code:

```bash
python onnxruntime-genai/examples/python/nemotron_speech.py \
    --model_path src/build/onnx_models_int4 \
    --audio_file path/to/audio.wav \
    --language de \
    -e cpu
```

Supported language codes match the NeMo multilingual prompt schema
(e.g. `en`, `de`, `fr`, `es`, `pt`, `it`, `nl`, `pl`, `zh-CN`, `ja-JP`, `auto`, …).
The full mapping is printed via `--help`.
