# mistralai-Ministral-3-3B-Instruct-2512 - Baseline PyTorch Evaluation

This folder contains the PyTorch baseline evaluation recipe for `mistralai/Ministral-3-3B-Instruct-2512-BF16` on AI2D. It evaluates the Hugging Face BF16 model without ONNX conversion or quantization.

## Setup

```bash
pip install -r requirements.txt
```

## Run baseline evaluation

```bash
python mistralai-Ministral-3-3B-Instruct-2512_pytorch_with_eval.py --device cuda --num_samples 100
```

The script defaults to `mistralai/Ministral-3-3B-Instruct-2512-BF16`, because the default FP8 checkpoint requires FP8 kernels that are not available in all PyTorch environments. To evaluate a local or alternate checkpoint, pass `--pytorch_model`.

```bash
python mistralai-Ministral-3-3B-Instruct-2512_pytorch_with_eval.py --pytorch_model /path/to/checkpoint --device cuda --num_samples 100
```

## Baseline results

Evaluated on AI2D with the default BF16 Hugging Face checkpoint.

| Model | Device | Precision | Samples | Accuracy | Latency (s/sample) |
|-------|--------|-----------|---------|----------|---------------------|
| `mistralai/Ministral-3-3B-Instruct-2512-BF16` | CUDA | FP16 | 500 | 74.20% (371/500) | 0.18 |

Command:

```bash
python mistralai-Ministral-3-3B-Instruct-2512_pytorch_with_eval.py --pytorch_model mistralai/Ministral-3-3B-Instruct-2512-BF16 --device cuda --num_samples 500
```

Latency is per-sample end-to-end inference time and excludes model loading.