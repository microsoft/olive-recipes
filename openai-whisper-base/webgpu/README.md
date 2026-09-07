# openai-whisper-base — WebGPU optimization

This folder contains Olive recipes for optimizing openai-whisper-base targeting the WebGPU EP.

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: INT8 precision by default
- Example recipe filename: whisper-base_webgpu_int8.json

## Setup

1) Install Olive:
   - pip install olive-ai
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (WebGPU build)
3) Run Olive to build/optimize the model
   - olive run --config whisper-base_webgpu_int8.json

Additional notes:
- Sets all MatMul nodes to 8-bit using k-quant.
- Pipeline: `ModelBuilder` (fp16) → `OnnxKQuantQuantization` (k-quant INT8 MatMul weights)
- WebGPU enables GPU-accelerated inference in web browsers.
- Ensure your browser supports WebGPU (Chrome 113+, Edge 113+).


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

This README was auto-generated for the WebGPU EP of openai-whisper-base.
