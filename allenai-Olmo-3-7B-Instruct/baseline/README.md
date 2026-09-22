# allenai-Olmo-3-7B-Instruct - Baseline PyTorch Evaluation

This folder contains an Olive recipe for evaluating the Hugging Face PyTorch base model `allenai/Olmo-3-7B-Instruct` with LM-eval MMLU.

## Setup

```bash
pip install -r requirements.txt
```

## Run evaluation

```bash
olive run --config allenai-Olmo-3-7B-Instruct_pytorch_with_eval.json
```

## Evaluation results (A100)

PyTorch baseline MMLU accuracy: `0.561814556331007`.
