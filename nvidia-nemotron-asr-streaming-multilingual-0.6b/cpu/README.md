# Nemotron 3.5 ASR Streaming Multilingual 0.6B — CPU

This recipe exports
**nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b** to ONNX for
CPU inference:

- **Encoder**: INT4 k-quant by default, or INT8 dynamic quantization
- **Decoder**: FP32
- **Joint**: FP32

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nvidia-nemotron-asr-streaming-multilingual-0.6b/cpu/requirements.txt
```

## Run

From the recipe directory:

```bash
cd nvidia-nemotron-asr-streaming-multilingual-0.6b

# INT4 encoder (default)
python cpu/optimize.py

# INT8 encoder
python cpu/optimize.py --encoder-precision int8

# Custom output directory
python cpu/optimize.py --output-dir build/multilingual_onnx_int4
```

The pipeline exports the encoder, decoder, joint network, tokenizer, and
runtime configuration files, and downloads Silero VAD.

Individual Olive configurations can also be run directly:

```bash
python -m olive run --config cpu/nemotron_encoder_int4_cpu.json
python -m olive run --config cpu/nemotron_encoder_int8_cpu.json
python -m olive run --config cpu/nemotron_decoder_fp32_cpu.json
python -m olive run --config cpu/nemotron_joint_fp32_cpu.json
```

## Output

The default output directory is `cpu/build/onnx_models_int4/`. It contains
`encoder.onnx`, `decoder.onnx`, `joint.onnx`, tokenizer files,
`genai_config.json`, `audio_processor_config.json`, and `silero_vad.onnx`.

## Inference

```bash
python onnxruntime-genai/examples/python/nemotron_speech.py \
    --model_path cpu/build/onnx_models_int4 \
    --audio_file path/to/audio.wav \
    --language de \
    -e cpu
```
