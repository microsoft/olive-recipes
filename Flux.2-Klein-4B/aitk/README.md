# FLUX.2-klein-4B — AMD NPU (VitisAI) recipe

This recipe optimizes [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
for the AMD Ryzen AI NPU. It converts each pipeline component to ONNX with Olive and compiles the
transformer and VAE decoder for the NPU with `VitisGenerateModelSD`; the text encoder is INT4
weight-quantized and the VAE encoder stays in plain ONNX.

| Component | Target | Optimization |
|---|---|---|
| Transformer (Flux2) | AMD NPU | `VitisGenerateModelSD` (`model_type: flux2`) |
| VAE decoder | AMD NPU | `VitisGenerateModelSD` (`model_type: sd3-vae-decoder`) |
| Text encoder (Qwen3) | CPU | ORT transformer opt + INT4 RTN block quantization |
| VAE encoder | CPU | ONNX conversion |

## How it runs

`flux_vitisai_workflow.json` is an `AitkPython` wrapper. At build time
`flux_vitisai_workflow.py` stages the per-component Olive configs
(`vitisai_config_*.json` → `config_*.json`) and delegates to `export_models.py`, which:

1. Loads the FLUX.2-klein-4B pipeline from HuggingFace (gated — accept the license first).
2. Exports each sub-model to ONNX via Olive.
3. Compiles the transformer and VAE decoder for the AMD NPU.
4. Assembles a self-contained pipeline directory (ONNX models + tokenizer + scheduler +
   `model_index.json`) under `model/flux_vitisai`.

The component Olive configs in this folder are kept in sync with the upstream AMD reference
configs in [`../RyzenAI`](../RyzenAI); sanitize diffs their `passes` blocks.

## Inference

See `flux_vitisai_workflow_inference_sample.ipynb`. The transformer and VAE decoder execute on the
AMD NPU through the VitisAI execution provider; the text encoder and VAE encoder run on CPU.

## Performance

> Measured on: _AMD Ryzen AI (NPU) — fill in device / driver / resolution and the numbers from your run._

| Metric | Value |
|---|---|
| Resolution | 1024×1024 |
| Inference steps | 28 |
| End-to-end latency | _TBD_ |
| Transformer latency / step | _TBD_ |

Numbers depend on the Ryzen AI device, driver, and runtime versions. Re-measure on your own
hardware before relying on them.
