# nvidia-Mistral-NeMo-12B-Instruct - Baseline PyTorch Evaluation

This folder contains an Olive recipe for evaluating `mistralai/Mistral-Nemo-Instruct-2407` (Hugging Face mirror of `nvidia/Mistral-NeMo-12B-Instruct`) with LM-eval MMLU as the PyTorch baseline.

> Note: The model is sourced from `mistralai/Mistral-Nemo-Instruct-2407`, the official Hugging Face mirror of `nvidia/Mistral-NeMo-12B-Instruct`. The original `nvidia/Mistral-NeMo-12B-Instruct` repo only ships a legacy NeMo 1.x `.nemo` archive (Megatron-Core distributed checkpoint plus a tekken tokenizer JSON), which has no Transformers `config.json`/`model_type` and is not consumable by `nemo.collections.llm.export_ckpt(target="hf")` in NeMo 2.x. Switching to the HF mirror lets the standard `HfModel` pipeline work end-to-end.

## Setup

```bash
pip install -r requirements.txt
```

## Run evaluation

```bash
olive run --config nvidia-Mistral-NeMo-12B-Instruct_pytorch_with_eval.json
```

## Evaluation results (A100)

- PyTorch baseline mmlu-acc: `0.6655747044580544` (humanities `0.6116896918172158`, stem `0.5645417063114494`, social_sciences `0.7760805979850504`, other `0.7402639201802381`).
