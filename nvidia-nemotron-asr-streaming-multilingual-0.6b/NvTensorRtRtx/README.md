# Nemotron 3.5 ASR Streaming Multilingual 0.6B — TRT-RTX

This recipe exports
**nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b** for the
NvTensorRtRtx execution provider. The encoder, decoder, and joint network use
homogeneous FP16 inputs, outputs, and internal compute at ONNX opset 23.

Silero VAD is omitted because it is not supported by TRT-RTX.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nvidia-nemotron-asr-streaming-multilingual-0.6b/NvTensorRtRtx/requirements.txt
```

## Run

From the recipe directory:

```bash
cd nvidia-nemotron-asr-streaming-multilingual-0.6b

python NvTensorRtRtx/optimize.py

# Custom output directory
python NvTensorRtRtx/optimize.py --output-dir build/multilingual_onnx_fp16
```

Individual Olive configurations can also be run directly:

```bash
python -m olive run --config NvTensorRtRtx/nemotron_encoder_fp16_trtrtx.json
python -m olive run --config NvTensorRtRtx/nemotron_decoder_fp16_trtrtx.json
python -m olive run --config NvTensorRtRtx/nemotron_joint_fp16_trtrtx.json
```

## Output

The default output directory is `NvTensorRtRtx/build/onnx_models_fp16/`.
It contains FP16 `encoder.onnx`, `decoder.onnx`, and `joint.onnx`
models, tokenizer files, `genai_config.json`, and
`audio_processor_config.json`.
