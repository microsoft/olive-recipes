# Qwen-Qwen3-Embedding-0.6B — CUDA optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-Embedding-0.6B targeting the CUDA EP.

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: INT4
- Recipe: `Qwen-Qwen3-Embedding-0.6B_cuda_int4.json` (build only), `Qwen-Qwen3-Embedding-0.6B_cuda_int4_with_eval.json` (build + evaluate)

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the requirements for this recipe:
   - pip install -r requirements.txt

## Build the model

```bash
olive run --config Qwen-Qwen3-Embedding-0.6B_cuda_int4.json
```

After building, copy `config_sentence_transformers.json` into the model output directory. This file provides task-specific query prompts required by MTEB retrieval benchmarks (e.g., NFCorpus). The GraphSurgeries pass does not carry it forward from the ModelBuilder output, so it must be copied manually:

```bash
cp ~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/config_sentence_transformers.json model_cuda_int4/
```

## Build and evaluate with MTEB

To build the model and run the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) STS17 benchmark comparing the source HuggingFace model against the exported ONNX/GenAI model:

```bash
olive run --config Qwen-Qwen3-Embedding-0.6B_cuda_int4_with_eval.json
```

> **Note:** Ensure `config_sentence_transformers.json` is present in the model output directory before running evaluation (see copy step above). Without it, retrieval benchmarks like NFCorpus will show ~20% lower scores.

The evaluation results will be logged at the end of the run, showing scores for both the source (HF) and exported (GenAI) models. The MTEB score of the exported ONNX model should be within 5% of the base PyTorch model.

## Evaluation results (A100)

### ONNX CUDA INT4

| Benchmark | Baseline | Score | Delta |
|-----------|----------|-------|-------|
| STS17 | 0.7852 | 0.7820 | -0.41% |
| NFCorpus | 0.3562 | 0.3507 | -1.54% |
| ArguAna | 0.6747 | 0.6725 | -0.33% |
| SciFact | 0.7015 | 0.6984 | -0.44% |

## Additional notes

- Pipeline: SelectiveMixedPrecision → GPTQ → RTN → ModelBuilder → GraphSurgeries (INT4 with include_hidden_states)
- This is an embedding model — outputs hidden states for embedding generation.
- Requires NVIDIA GPU with CUDA support.
- Ensure CUDA toolkit and cuDNN are properly installed.
