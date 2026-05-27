# Qwen3-VL-2B-Instruct Vision Evaluation

Evaluate quantized ONNX Qwen3-VL-2B-Instruct on vision benchmarks using Olive's built-in `exact_match` metric.

## Benchmarks

- **AI2D** — Science diagram multiple-choice QA (`lmms-lab/ai2d`, test split)

## Prerequisites

```bash
pip install -r requirements.txt
```

Ensure you have the ONNX model built (see `../cpu_and_mobile/` or `../cuda/`).

## Usage

### Standalone evaluation

```bash
# Evaluate ONNX model on 100 samples (default)
python evaluate.py --model_path ../cpu_and_mobile/models

# Full dataset
python evaluate.py --model_path ../cpu_and_mobile/models --num_samples 0

# Compare ONNX vs PyTorch
python evaluate.py --model_path ../cpu_and_mobile/models --pytorch_model Qwen/Qwen3-VL-2B-Instruct
```

### Via Olive (integrated evaluation)

```bash
olive run --config qwen3vl_eval_ai2d.json
```

## Metrics

| Metric | Description | Task |
|--------|-------------|------|
| `exact_match` | Case-insensitive string equality | VQA (AI2D, ScienceQA, TextVQA) |
| `relaxed_accuracy` | ±5% numeric tolerance | ChartQA |
| `word_sort_ratio` | Word-level overlap ratio | OCR |

## Related

- Olive vision metrics PR: https://github.com/microsoft/Olive/pull/2474
- Speech evaluation recipe: `nvidia-nemotron-speech-streaming-en-0.6b/cpu/`
