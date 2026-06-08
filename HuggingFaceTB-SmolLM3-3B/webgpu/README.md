# HuggingFaceTB-SmolLM3-3B - WebGPU Optimization

This folder contains Olive recipes for optimizing `HuggingFaceTB/SmolLM3-3B` for `WebGpuExecutionProvider`.

## Recipes

- `HuggingFaceTB-SmolLM3-3B_webgpu_fp16.json`
- `HuggingFaceTB-SmolLM3-3B_webgpu_fp16_with_eval.json`
- `HuggingFaceTB-SmolLM3-3B_webgpu_int4.json`
- `HuggingFaceTB-SmolLM3-3B_webgpu_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config HuggingFaceTB-SmolLM3-3B_webgpu_fp16.json
olive run --config HuggingFaceTB-SmolLM3-3B_webgpu_int4.json
```

## Build and evaluate with MMLU

```bash
olive run --config HuggingFaceTB-SmolLM3-3B_webgpu_fp16_with_eval.json
olive run --config HuggingFaceTB-SmolLM3-3B_webgpu_int4_with_eval.json
```

## Notes

- SmolLM3 config has tie_word_embeddings=true, so TieWordEmbeddings surgery is applied after ModelBuilder.
- Full precision recipe for this backend uses `fp16`.
- INT4 recipes follow the Qwen-Qwen3-4B pass chain: SelectiveMixedPrecision -> GPTQ -> RTN -> ModelBuilder.

## Evaluation results (A100)

The `WebGpuExecutionProvider` is not available on the headless Linux A100 VM (it requires a Vulkan/Dawn surface). Because the WebGPU INT4 recipe produces the exact same ONNX graph that ORT runs on every EP, we evaluate the artifact via `CUDAExecutionProvider` as an accuracy parity check; quantization fidelity is decided at build time, so CUDA-EP MMLU numbers are a faithful proxy for what WebGPU will produce in the browser.

- PyTorch baseline accuracy: `0.5931491240564022`
- WebGPU INT4 accuracy (via CUDA EP): `0.589588377723971`
- Delta vs. baseline: `-0.0035607463324312`
