# nvidia-Mistral-NeMo-12B-Instruct - WEBGPU Optimization

This folder contains Olive recipes for optimizing `mistralai/Mistral-Nemo-Instruct-2407` (Hugging Face mirror of `nvidia/Mistral-NeMo-12B-Instruct`) for `WebGpuExecutionProvider`.

> Note: The model is sourced from `mistralai/Mistral-Nemo-Instruct-2407`, the official Hugging Face mirror of `nvidia/Mistral-NeMo-12B-Instruct`. The original `nvidia/Mistral-NeMo-12B-Instruct` repo only ships a legacy NeMo 1.x `.nemo` archive (Megatron-Core distributed checkpoint plus a tekken tokenizer JSON), which has no Transformers `config.json`/`model_type` and is not consumable by `nemo.collections.llm.export_ckpt(target="hf")` in NeMo 2.x. Switching to the HF mirror lets the standard `HfModel` pipeline work end-to-end.

## Recipes

- `nvidia-Mistral-NeMo-12B-Instruct_webgpu_fp16.json`
- `nvidia-Mistral-NeMo-12B-Instruct_webgpu_fp16_with_eval.json`
- `nvidia-Mistral-NeMo-12B-Instruct_webgpu_int4.json`
- `nvidia-Mistral-NeMo-12B-Instruct_webgpu_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config nvidia-Mistral-NeMo-12B-Instruct_webgpu_fp16.json
olive run --config nvidia-Mistral-NeMo-12B-Instruct_webgpu_int4.json
```

## Build and evaluate with MMLU

```bash
olive run --config nvidia-Mistral-NeMo-12B-Instruct_webgpu_fp16_with_eval.json
olive run --config nvidia-Mistral-NeMo-12B-Instruct_webgpu_int4_with_eval.json
```

## Notes

- `input_model` uses `HfModel` with `model_path` set to `mistralai/Mistral-Nemo-Instruct-2407`.
- TieWordEmbeddings surgery is intentionally omitted. If the HF config later exposes `tie_word_embeddings=true`, add the usual `GraphSurgeries` `TieWordEmbeddings` pass after ModelBuilder.
- Full precision recipe for this backend uses `fp16`.
- INT4 recipes follow the standard pass chain: SelectiveMixedPrecision -> GPTQ -> RTN -> ModelBuilder.

## Evaluation results (A100)

The `WebGpuExecutionProvider` is not available on the headless Linux A100 VM (it requires a Vulkan/Dawn surface). Because the WebGPU INT4 recipe produces the exact same ONNX graph that ORT runs on every EP, we evaluate the artifact via `CUDAExecutionProvider` as an accuracy parity check; quantization fidelity is decided at build time, so CUDA-EP MMLU numbers are a faithful proxy for what WebGPU will produce in the browser.

- PyTorch baseline accuracy: `0.6655747044580544`
- WebGPU INT4 accuracy (via CUDA EP): `0.6484119071357356`
- Delta vs. baseline: `-0.0171627973223188`
