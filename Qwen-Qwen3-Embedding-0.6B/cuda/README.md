# Qwen-Qwen3-Embedding-0.6B — CUDA optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-Embedding-0.6B targeting the CUDA EP.

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: FP32
- Example recipe filename: Qwen-Qwen3-Embedding-0.6B_cuda_fp32.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai-cuda (CUDA build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-Embedding-0.6B_cuda_fp32.json

Additional notes:
- Pipeline: `ModelBuilder` (fp32 with include_hidden_states)
- This is an embedding model — outputs hidden states for embedding generation.
- Requires NVIDIA GPU with CUDA support.
- Ensure CUDA toolkit and cuDNN are properly installed.

## MTEB Benchmark (source model)

Scores for [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (0.6B params, 32K context, up to 1024-dim embeddings):

| Benchmark | Mean (Task) |
|-----------|-------------|
| MTEB Multilingual | 64.33 |
| MTEB English v2 | 70.70 |
| C-MTEB (Chinese) | 66.33 |

Source: [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), May 24, 2025.

---

This README was auto-generated for the CUDA EP of Qwen-Qwen3-Embedding-0.6B.
