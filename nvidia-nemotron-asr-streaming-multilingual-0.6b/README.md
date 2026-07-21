# nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b

Olive recipes for [nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b),
a 0.6B-parameter streaming multilingual ASR model covering 100+ languages.

## Performance

Multilingual evaluation is in progress on **FLEURS**, **Common Voice**,
**Multilingual LibriSpeech (MLS)**, and **VoxPopuli**. Per-language WER
numbers will be published here once the matrix is complete.

## Recipes

- [`cpu/`](./cpu) — INT4 or INT8 encoder with FP32 decoder and joint models for
  the CPU execution provider.
- [`NvTensorRtRtx/`](./NvTensorRtRtx) — homogeneous FP16 opset-23 encoder,
  decoder, and joint models for the NvTensorRtRtx execution provider.

See the README inside each subfolder for setup and run instructions.

## Inference

Streaming inference is supported via [`onnxruntime-genai`](https://github.com/microsoft/onnxruntime-genai),
with a per-utterance `--language` flag (or `auto`) that selects the encoder
prompt token. For example, after running the CPU recipe:

```bash
python onnxruntime-genai/examples/python/nemotron_speech.py \
    --model_path cpu/build/onnx_models_int4 \
    --audio_file path/to/audio.wav \
    --language de \
    -e cpu
```
