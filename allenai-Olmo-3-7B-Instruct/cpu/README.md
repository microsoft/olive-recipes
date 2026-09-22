# allenai-Olmo-3-7B-Instruct - CPU Optimization

This folder contains Olive recipes for optimizing `allenai/Olmo-3-7B-Instruct` for `CPUExecutionProvider`.

## Recipes

- `allenai-Olmo-3-7B-Instruct_cpu_fp32.json`
- `allenai-Olmo-3-7B-Instruct_cpu_fp32_with_eval.json`
- `allenai-Olmo-3-7B-Instruct_cpu_int4.json`
- `allenai-Olmo-3-7B-Instruct_cpu_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config allenai-Olmo-3-7B-Instruct_cpu_fp32.json
olive run --config allenai-Olmo-3-7B-Instruct_cpu_int4.json
```

## Build and evaluate with MMLU

```bash
olive run --config allenai-Olmo-3-7B-Instruct_cpu_fp32_with_eval.json
olive run --config allenai-Olmo-3-7B-Instruct_cpu_int4_with_eval.json
```

## Notes

- OLMo-3 config has tie_word_embeddings=false, so TieWordEmbeddings surgery is intentionally omitted.
- Full precision recipe for this backend uses `fp32`.
- INT4 recipes follow the Qwen-Qwen3-4B pass chain: SelectiveMixedPrecision -> GPTQ -> RTN -> ModelBuilder.

## Evaluation results (A100)

MMLU was run for the CPU INT4 recipe only, with fp32 eval recipes kept but not executed. The full MMLU run through the direct ORT CPU evaluator was projected at roughly `17-28` hours on `CPUExecutionProvider`, so a sampled MMLU evaluation with `limit=256` per subject was used instead.

- PyTorch baseline accuracy (full MMLU): `0.561814556331007`
- CPU INT4 accuracy (sampled MMLU, `limit=256`): `0.5844886088221037`
  - humanities: `0.5712621359223301`
  - stem: `0.533580830239622`
  - social_sciences: `0.6583986074847694`
  - other: `0.590560709963695`
- Multi-turn repetition check: passed; no repeated 4-grams or same-word loops were observed.
