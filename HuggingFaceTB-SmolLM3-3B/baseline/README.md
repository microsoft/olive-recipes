# HuggingFaceTB-SmolLM3-3B - Baseline PyTorch Evaluation

This folder contains an Olive recipe for evaluating the Hugging Face PyTorch base model `HuggingFaceTB/SmolLM3-3B` with LM-eval MMLU.

## Setup

```bash
pip install -r requirements.txt
```

## Run evaluation

```bash
olive run --config HuggingFaceTB-SmolLM3-3B_pytorch_with_eval.json
```

## Evaluation results (A100)

PyTorch baseline MMLU accuracy: `0.5931491240564022`.
