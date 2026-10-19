# Qwen2.5-VL-3B-Instruct Vision Evaluation

Evaluate quantized ONNX Qwen2.5-VL-3B-Instruct on vision benchmarks using Olive's built-in `exact_match` metric.

## Benchmarks

- **AI2D** — Science diagram multiple-choice QA (`lmms-lab/ai2d`, test split)

## Prerequisites

Ensure you have the ONNX model built (see `../cpu_and_mobile/` or `../cuda/`).

## Usage

```bash
# CPU evaluation
olive run --config ai2d_cpu.json

# CUDA evaluation
olive run --config ai2d_cuda.json
```

## Configs

| Config | Model Path | Device |
|--------|-----------|--------|
| `ai2d_cpu.json` | `../cpu_and_mobile/models` | CPU |
| `ai2d_cuda.json` | `../cuda/models` | GPU (CUDA) |
