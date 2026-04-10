# Qwen-Qwen3-Embedding-0.6B — WebGPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-Embedding-0.6B targeting the WebGPU EP.

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: FP32
- Example recipe filename: Qwen-Qwen3-Embedding-0.6B_webgpu_fp32.json

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-web (WebGPU build)
3) Run Olive to build/optimize the model
   - olive run --config Qwen-Qwen3-Embedding-0.6B_webgpu_fp32.json

Additional notes:
- Pipeline: `ModelBuilder` (fp32 with include_hidden_states)
- This is an embedding model — outputs hidden states for embedding generation.
- WebGPU enables GPU-accelerated inference in web browsers.
- Ensure your browser supports WebGPU (Chrome 113+, Edge 113+).

## MTEB Benchmark (source model)

Scores for [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (0.6B params, 32K context, up to 1024-dim embeddings):

| Benchmark | Mean (Task) |
|-----------|-------------|
| MTEB Multilingual | 64.33 |
| MTEB English v2 | 70.70 |
| C-MTEB (Chinese) | 66.33 |

Source: [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), May 24, 2025.

---

This README was auto-generated for the WebGPU EP of Qwen-Qwen3-Embedding-0.6B.
