# Qwen-Qwen3-Embedding-0.6B — Baseline PyTorch Evaluation

This folder contains the baseline evaluation recipe for the Qwen3-Embedding-0.6B model. It evaluates the original HuggingFace PyTorch model without any ONNX conversion or quantization.

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the requirements for this recipe:
   - pip install -r requirements.txt

## Run baseline evaluation

```bash
olive run --config Qwen-Qwen3-Embedding-0.6B_pytorch_with_eval.json
```

## Baseline results (A100)

| Benchmark | Score |
|-----------|-------|
| STS17 | 0.7852 |
| NFCorpus | 0.3562 |
| ArguAna | 0.6747 |
| SciFact | 0.7015 |
