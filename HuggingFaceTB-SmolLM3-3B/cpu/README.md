# HuggingFaceTB-SmolLM3-3B - CPU Optimization

This folder contains Olive recipes for optimizing `HuggingFaceTB/SmolLM3-3B` for `CPUExecutionProvider`.

## Recipes

- `HuggingFaceTB-SmolLM3-3B_cpu_fp32.json`
- `HuggingFaceTB-SmolLM3-3B_cpu_fp32_with_eval.json`
- `HuggingFaceTB-SmolLM3-3B_cpu_int4.json`
- `HuggingFaceTB-SmolLM3-3B_cpu_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config HuggingFaceTB-SmolLM3-3B_cpu_fp32.json
olive run --config HuggingFaceTB-SmolLM3-3B_cpu_int4.json
```

## Build and evaluate with MMLU

```bash
olive run --config HuggingFaceTB-SmolLM3-3B_cpu_fp32_with_eval.json
olive run --config HuggingFaceTB-SmolLM3-3B_cpu_int4_with_eval.json
```

## Notes

- SmolLM3 config has tie_word_embeddings=true, so TieWordEmbeddings surgery is applied after ModelBuilder.
- Full precision recipe for this backend uses `fp32`.
- INT4 recipes follow the Qwen-Qwen3-4B pass chain: SelectiveMixedPrecision -> GPTQ -> RTN -> ModelBuilder.

## Evaluation results (A100)

The CPU INT4 evaluation path was run, with fp32 eval recipes kept but not executed. The direct ORT CPU MMLU path was projected at roughly `17-28` hours for the full `56168` loglikelihood requests, so a sampled MMLU evaluation with `limit=256` per subject was used instead.

- PyTorch baseline accuracy (full MMLU): `0.5931491240564022`
- CPU INT4 accuracy (sampled MMLU, `limit=256`): `0.6159961221522056`
  - humanities: `0.6264077669902912`
  - stem: `0.5474181572730341`
  - social_sciences: `0.6940818102697999`
  - other: `0.6147640177490924`
- Multi-turn repetition check: passed; no repeated 4-grams or same-word loops were observed.
