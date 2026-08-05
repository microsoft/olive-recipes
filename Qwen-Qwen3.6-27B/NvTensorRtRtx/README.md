# Qwen3.6-27B optimization

This folder contains an Olive recipe for exporting the text-only component of `Qwen/Qwen3.6-27B` for the
`NvTensorRTRTXExecutionProvider` (also known as the `NvTensorRtRtx` EP).

## INT4 weight-only quantization

The `Qwen3.6-27B_model_builder_int4.json` recipe uses the ONNX Runtime GenAI Qwen3.5 hybrid model builder, which
matches the architecture declared by the Qwen3.6 checkpoint, to:

1. Export a standalone text model by including the token embedding layer (`exclude_embeds=false`).
2. Apply symmetric INT4 weight-only quantization with a block size of 32 using ModelBuilder's default quantizer.
3. Emit INT4 QDQ directly for `NvTensorRTRTXExecutionProvider`, and run `MatMulNBitsToQDQ` as a compatibility fallback for any remaining `MatMulNBits` nodes.
4. Enable the shared past/present buffer and CUDA graph capture for TRT-RTX inference.

The vision encoder is not exported.

The `export.py` entry point runs Olive and validates that the final model contains signed INT4 QDQ weights, no
`MatMulNBits` nodes, a shared past/present buffer, and CUDA graph configuration. The command fails instead of
silently returning an incompatible model if any requirement is missing.

## Setup

1. Install Olive.
2. Install a Transformers 5.x release that recognizes the `qwen3_5` architecture.
3. Install an ONNX Runtime GenAI package or build that supports Qwen3.5 hybrid models and the
   `NvTensorRTRTXExecutionProvider`.

## Run

```bash
python export.py -o output
```

The TRT-RTX-ready model is written to `output/model.onnx`.
