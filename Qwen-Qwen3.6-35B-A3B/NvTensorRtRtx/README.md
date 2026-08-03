# Qwen3.6-35B-A3B optimization

This folder contains an Olive recipe for exporting the text-only component of `Qwen/Qwen3.6-35B-A3B` for the
`NvTensorRTRTXExecutionProvider` (also known as the `NvTensorRtRtx` EP).

## INT4 weight-only quantization

The `Qwen3.6-35B-A3B_model_builder_int4.json` recipe uses the ONNX Runtime GenAI Qwen3.5 MoE hybrid model builder,
which matches the architecture declared by the Qwen3.6 checkpoint, to:

1. Export a standalone text model by including the token embedding layer (`exclude_embeds=false`).
2. Apply symmetric INT4 RTN weight-only quantization with a block size of 32.
3. Export quantized matrix multiplications directly in INT4 QDQ format (`use_qdq=true`).

The vision encoder is not exported.

## Setup

1. Install Olive.
2. Install a Transformers 5.x release that recognizes the `qwen3_5_moe` architecture.
3. Install an ONNX Runtime GenAI package or build that supports Qwen3.5 MoE hybrid models and the
   `NvTensorRTRTXExecutionProvider`.

## Run

```bash
olive run --config Qwen3.6-35B-A3B_model_builder_int4.json
```
