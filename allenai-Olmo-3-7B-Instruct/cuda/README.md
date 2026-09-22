# allenai-Olmo-3-7B-Instruct - CUDA Optimization

This folder contains Olive recipes for optimizing `allenai/Olmo-3-7B-Instruct` for `CUDAExecutionProvider`.

## Recipes

- `allenai-Olmo-3-7B-Instruct_cuda_fp16.json`
- `allenai-Olmo-3-7B-Instruct_cuda_fp16_with_eval.json`
- `allenai-Olmo-3-7B-Instruct_cuda_int4.json`
- `allenai-Olmo-3-7B-Instruct_cuda_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config allenai-Olmo-3-7B-Instruct_cuda_fp16.json
olive run --config allenai-Olmo-3-7B-Instruct_cuda_int4.json
```

## Build and evaluate with MMLU

```bash
olive run --config allenai-Olmo-3-7B-Instruct_cuda_fp16_with_eval.json
olive run --config allenai-Olmo-3-7B-Instruct_cuda_int4_with_eval.json
```

## Notes

- OLMo-3 config has tie_word_embeddings=false, so TieWordEmbeddings surgery is intentionally omitted.
- Full precision recipe for this backend uses `fp16`.
- INT4 recipes follow the Qwen-Qwen3-4B pass chain: SelectiveMixedPrecision -> GPTQ -> RTN -> ModelBuilder.

## Evaluation results (A100)

MMLU was run for the CUDA INT4 recipe only, with fp16 eval recipes kept but not executed.

- PyTorch baseline accuracy: `0.561814556331007`
- CUDA INT4 accuracy: `0.5613872667711153`
- Delta vs. baseline: `-0.0004272895598917`
- Multi-turn repetition check: passed; no repeated 4-grams or same-word loops were observed.
