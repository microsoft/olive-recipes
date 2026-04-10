# Qwen-Qwen3-Embedding-0.6B — CPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-Embedding-0.6B targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: FP32
- Example recipe filename: Qwen-Qwen3-Embedding-0.6B_cpu_fp32.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai (CPU build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-Embedding-0.6B_cpu_fp32.json

Additional notes:
- Pipeline: `ModelBuilder` (fp32 with include_hidden_states)
- This is an embedding model — outputs hidden states for embedding generation.
- Runs purely on CPU; no GPU required.

## MTEB Benchmark (source model)

Scores for [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (0.6B params, 32K context, up to 1024-dim embeddings):

| Benchmark | Mean (Task) |
|-----------|-------------|
| MTEB Multilingual | 64.33 |
| MTEB English v2 | 70.70 |
| C-MTEB (Chinese) | 66.33 |

Source: [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), May 24, 2025.

## STS17 Evaluation (Olive MTEBEvaluator)

Scores from running the MTEB STS17 task via the Olive `MTEBEvaluator`, comparing the source HuggingFace model against the exported ONNX/GenAI model:

| Subset | Source (HF) | Exported (GenAI) |
|--------|-------------|------------------|
| **main_score** | **0.785** | **0.681** |
| ko-ko | 0.763 | 0.752 |
| ar-ar | 0.752 | 0.684 |
| en-ar | 0.700 | 0.563 |
| en-de | 0.846 | 0.688 |
| en-en | 0.907 | 0.834 |
| en-tr | 0.585 | 0.437 |
| es-en | 0.804 | 0.719 |
| es-es | 0.861 | 0.792 |
| fr-en | 0.811 | 0.705 |
| it-en | 0.832 | 0.728 |
| nl-en | 0.775 | 0.593 |

---

This README was auto-generated for the CPU EP of Qwen-Qwen3-Embedding-0.6B.
