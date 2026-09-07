# nvidia-Mistral-NeMo-12B-Instruct - CPU Optimization

This folder contains Olive recipes for optimizing `mistralai/Mistral-Nemo-Instruct-2407` (Hugging Face mirror of `nvidia/Mistral-NeMo-12B-Instruct`) for `CPUExecutionProvider`.

> Note: The model is sourced from `mistralai/Mistral-Nemo-Instruct-2407`, the official Hugging Face mirror of `nvidia/Mistral-NeMo-12B-Instruct`. The original `nvidia/Mistral-NeMo-12B-Instruct` repo only ships a legacy NeMo 1.x `.nemo` archive (Megatron-Core distributed checkpoint plus a tekken tokenizer JSON), which has no Transformers `config.json`/`model_type` and is not consumable by `nemo.collections.llm.export_ckpt(target="hf")` in NeMo 2.x. Switching to the HF mirror lets the standard `HfModel` pipeline work end-to-end.

## Recipes

- `nvidia-Mistral-NeMo-12B-Instruct_cpu_fp32.json`
- `nvidia-Mistral-NeMo-12B-Instruct_cpu_fp32_with_eval.json`
- `nvidia-Mistral-NeMo-12B-Instruct_cpu_int4.json`
- `nvidia-Mistral-NeMo-12B-Instruct_cpu_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config nvidia-Mistral-NeMo-12B-Instruct_cpu_fp32.json
olive run --config nvidia-Mistral-NeMo-12B-Instruct_cpu_int4.json
```

## Build and evaluate with MMLU

```bash
olive run --config nvidia-Mistral-NeMo-12B-Instruct_cpu_fp32_with_eval.json
olive run --config nvidia-Mistral-NeMo-12B-Instruct_cpu_int4_with_eval.json
```

## Notes

- `input_model` uses `HfModel` with `model_path` set to `mistralai/Mistral-Nemo-Instruct-2407`.
- TieWordEmbeddings surgery is intentionally omitted. If the HF config later exposes `tie_word_embeddings=true`, add the usual `GraphSurgeries` `TieWordEmbeddings` pass after ModelBuilder.
- Full precision recipe for this backend uses `fp32`.
- INT4 recipes follow the standard pass chain: SelectiveMixedPrecision -> GPTQ -> RTN -> ModelBuilder.

## Evaluation results (A100)

- PyTorch baseline accuracy: `0.6655747044580544`
- CPU INT4 accuracy: `0.6472012533827091` (humanities `0.59192348565356`, stem `0.5461465271170314`, social_sciences `0.7552811179720507`, other `0.7264242034116511`)
- Delta vs. baseline: `-0.0183734510753453`
- Repetition heuristic (>=3 four-gram repeats over a 5-prompt sweep) passed.
