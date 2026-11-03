# Qwen3.6-35B-A3B optimization

This folder contains an Olive recipe for exporting the text-only component of `Qwen/Qwen3.6-35B-A3B` for the
`NvTensorRTRTXExecutionProvider` (also known as the `NvTensorRtRtx` EP).

## INT4 weight-only quantization

The `Qwen3.6-35B-A3B_NvTensorRtRtx.json` recipe uses the ONNX Runtime GenAI Qwen3.5 MoE hybrid model builder,
which matches the architecture declared by the Qwen3.6 checkpoint, to:

1. Export a standalone text model by including the token embedding layer (`exclude_embeds=false`).
2. Apply symmetric INT4 RTN weight-only quantization with a block size of 32.
3. Convert any resulting `MatMulNBits` nodes to signed INT4 QDQ with the explicit `MatMulNBitsToQDQ` pass.
4. Enable the shared past/present buffer and CUDA graph capture for TRT-RTX inference.

The vision encoder is not exported.
