# Qwen-Qwen3-Embedding-8B — Baseline PyTorch Evaluation

This folder contains the baseline evaluation recipe for the Qwen3-Embedding-8B model. It evaluates the original HuggingFace PyTorch model without any ONNX conversion or quantization.

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the requirements for this recipe:
   - pip install -r requirements.txt
3) Log in to Hugging Face (the 8B model is gated and requires authentication):
   - huggingface-cli login

## Run baseline evaluation

```bash
olive run --config Qwen-Qwen3-Embedding-8B_pytorch_with_eval.json
```

## Baseline results (A100)

| Benchmark | Score |
|-----------|-------|
| STS17 | 0.8589 |
| NFCorpus | 0.4133 |
| ArguAna | 0.7539 |
| SciFact | 0.7859 |
