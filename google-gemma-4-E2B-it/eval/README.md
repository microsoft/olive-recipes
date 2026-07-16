# Gemma 4 E2B Evaluation

Evaluate quantized ONNX Gemma 4 E2B models on text, vision and audio benchmarks using Olive's built-in metrics.

## Benchmarks

- **MMLU Pro** — Multi-task language understanding (`leaderboard_mmlu_pro`, accuracy)
- **AI2D** — Science diagram multiple-choice QA (`lmms-lab/ai2d`, test split, exact_match)
- **FLEURS ASR** — Speech transcription accuracy (`google/fleurs`, en_us test split, WER + RTFx)

## Prerequisites

Ensure you have the ONNX models built using the mixed configs (see `../cpu/mixed/` or `../cuda/mixed/`).

## Usage

```bash
# Text evaluation (MMLU Pro)
olive run --config mmlu_cpu.json      # CPU
olive run --config mmlu_cuda.json     # CUDA

# Vision evaluation
olive run --config ai2d_cpu.json      # CPU
olive run --config ai2d_cuda.json     # CUDA

# Audio evaluation (FLEURS ASR)
olive run --config fleurs_asr_cpu.json   # CPU
olive run --config fleurs_asr_cuda.json  # CUDA
```

## Configs

### Text (MMLU Pro)

| Config | Model Path | Device |
|--------|-----------|--------|
| `mmlu_cpu.json` | `../cpu/mixed/models` | CPU |
| `mmlu_cuda.json` | `../cuda/mixed/models` | GPU (CUDA) |

These use Olive's `LMEvaluator` with the `ortgenai` backend. Gemma 4 exports with
`past_present_share_buffer` enabled, but the decoder requires it **disabled** for correct
KV-cache handling during evaluation, so the configs pass
`"model_args": { "past_present_share_buffer": false }`. Both `model_args` and the
`past_present_share_buffer` override require Olive with
[microsoft/Olive#2569](https://github.com/microsoft/Olive/pull/2569).
The default `limit` is 100 samples; remove it to run the full benchmark.
The configs also set `"sample_log_num": 100`, which writes the per-question prediction
vs. target for the first 100 samples to `sample_logs/leaderboard_mmlu_pro_samples.jsonl`
for inspection/debugging (set to `0` to disable).

### Vision (AI2D)

| Config | Model Path | Device |
|--------|-----------|--------|
| `ai2d_cpu.json` | `../cpu/mixed/models` | CPU |
| `ai2d_cuda.json` | `../cuda/mixed/models` | GPU (CUDA) |

### Audio (FLEURS ASR)

| Config | Model Path | Device |
|--------|-----------|--------|
| `fleurs_asr_cpu.json` | `../cpu/mixed/models/decoder` | CPU |
| `fleurs_asr_cuda.json` | `../cuda/mixed/models/decoder` | GPU (CUDA) |
