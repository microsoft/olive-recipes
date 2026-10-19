# HuggingFaceTB-SmolLM3-3B - CUDA Optimization

This folder contains Olive recipes for optimizing `HuggingFaceTB/SmolLM3-3B` for `CUDAExecutionProvider`.

## Recipes

- `HuggingFaceTB-SmolLM3-3B_cuda_fp16.json`
- `HuggingFaceTB-SmolLM3-3B_cuda_fp16_with_eval.json`
- `HuggingFaceTB-SmolLM3-3B_cuda_int4.json`
- `HuggingFaceTB-SmolLM3-3B_cuda_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config HuggingFaceTB-SmolLM3-3B_cuda_fp16.json
olive run --config HuggingFaceTB-SmolLM3-3B_cuda_int4.json
```

## Build and evaluate with MMLU

```bash
olive run --config HuggingFaceTB-SmolLM3-3B_cuda_fp16_with_eval.json
olive run --config HuggingFaceTB-SmolLM3-3B_cuda_int4_with_eval.json
```

## Notes

- SmolLM3 config has tie_word_embeddings=true, so TieWordEmbeddings surgery is applied after ModelBuilder.
- Full precision recipe for this backend uses `fp16`.
- INT4 recipes follow the Qwen-Qwen3-4B pass chain: SelectiveMixedPrecision -> GPTQ -> RTN -> ModelBuilder.

## Evaluation results (A100)

MMLU was run for the CUDA INT4 recipe only, with fp16 eval recipes kept but not executed.

- PyTorch baseline accuracy: `0.5931491240564022`
- CUDA INT4 accuracy: `0.5925081897165646`
- Delta vs. baseline: `-0.0006409343398376`
- Multi-turn repetition check: passed; no repeated 4-grams or same-word loops were observed.
