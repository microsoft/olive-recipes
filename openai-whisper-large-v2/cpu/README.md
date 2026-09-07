# openai-whisper-large-v2 — CPU optimization

This folder contains Olive recipes for optimizing openai-whisper-large-v2 targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: INT8 precision by default
- Example recipe filename: whisper-large-v2_cpu_int8.json

## Setup

1) Install Olive:
   - pip install olive-ai
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai
3) Run Olive to build/optimize the model
   - olive run --config whisper-large-v2_cpu_int8.json

Additional notes:
- Sets all MatMul nodes to 8-bit using k-quant.
- Pipeline: `ModelBuilder` (fp32) → `OnnxKQuantQuantization` (k-quant INT8 MatMul weights)
- Runs purely on CPU; no GPU required.


## Evaluation (WER & RTFx)

After exporting the model, you can evaluate transcription accuracy using Olive's built-in WER and RTFx metrics:

```bash
python -m olive run --config whisper_eval_wer.json
```

This evaluates the exported model on LibriSpeech test.clean (64 samples by default).

To change dataset or sample count, edit `whisper_eval_wer.json`:
- `max_samples`: Number of samples to evaluate (set to `0` for all)
- `subset`/`split`: Dataset subset and split (e.g., `"voxpopuli"` / `"test"`)

---

This README was auto-generated for the CPU EP of openai-whisper-large-v2.
