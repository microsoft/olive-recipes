# Qwen-Qwen3-Embedding-8B — CPU optimization

This folder contains Olive recipes for optimizing Qwen-Qwen3-Embedding-8B targeting the CPU EP.

## What this folder is for

- Execution Provider: CPU EP
- Typical precision: FP32
- Recipe: `Qwen-Qwen3-Embedding-8B_cpu_fp32.json` (build only), `Qwen-Qwen3-Embedding-8B_cpu_fp32_with_eval.json` (build + evaluate)

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the requirements for this recipe:
   - pip install -r requirements.txt
3) Log in to Hugging Face (the 8B model is gated and requires authentication):
   - huggingface-cli login

## Build the model

```bash
olive run --config Qwen-Qwen3-Embedding-8B_cpu_fp32.json
```

## Build and evaluate with MTEB

To build the model and run the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) STS17 benchmark comparing the source HuggingFace model against the exported ONNX/GenAI model:

```bash
olive run --config Qwen-Qwen3-Embedding-8B_cpu_fp32_with_eval.json
```

The evaluation results will be logged at the end of the run, showing scores for both the source (HF) and exported (GenAI) models. The MTEB score of the exported ONNX model should be within 5% of the base PyTorch model.

## Additional notes

- Pipeline: `ModelBuilder` (fp32 with include_hidden_states)
- This is an embedding model — outputs hidden states for embedding generation.
- Runs purely on CPU; no GPU required.
