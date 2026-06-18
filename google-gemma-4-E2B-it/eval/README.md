# Gemma 4 E2B Evaluation

Evaluate quantized ONNX Gemma 4 E2B models on vision and audio benchmarks using Olive's built-in metrics.

## Benchmarks

- **AI2D** — Science diagram multiple-choice QA (`lmms-lab/ai2d`, test split, exact_match)
- **Audio WER** — Speech transcription accuracy (`hf-audio/esb-datasets-test-only-sorted`, LibriSpeech test.clean, WER + RTFx)

## Prerequisites

Ensure you have the ONNX models built using the mixed configs (see `../cpu/mixed/` or `../cuda/mixed/`).

## Usage

```bash
# Vision evaluation
olive run --config ai2d_cpu.json      # CPU
olive run --config ai2d_cuda.json     # CUDA

# Audio evaluation
olive run --config audio_wer_cpu.json   # CPU
olive run --config audio_wer_cuda.json  # CUDA
```

## Configs

### Vision (AI2D)

| Config | Model Path | Device |
|--------|-----------|--------|
| `ai2d_cpu.json` | `../cpu/mixed/models` | CPU |
| `ai2d_cuda.json` | `../cuda/mixed/models` | GPU (CUDA) |

### Audio (WER)

| Config | Model Path | Device |
|--------|-----------|--------|
| `audio_wer_cpu.json` | `../cpu/mixed/models` | CPU |
| `audio_wer_cuda.json` | `../cuda/mixed/models` | GPU (CUDA) |
