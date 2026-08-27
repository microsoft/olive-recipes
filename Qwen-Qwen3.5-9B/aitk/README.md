# Qwen3.5-9B optimization

This folder contains an Olive recipe for exporting the text-only component of `Qwen/Qwen3.5-9B` for the
`NvTensorRTRTXExecutionProvider` (also known as the `NvTensorRtRtx` EP).

## INT4 weight-only quantization

The `Qwen3.5-0.8B_NvTensorRtRtx.json` recipe uses the ONNX Runtime GenAI `ModelBuilder` to:

1. Export a standalone text model by including the token embedding layer (`exclude_embeds=false`).
2. Apply symmetric INT4 RTN weight-only quantization with a block size of 32.
3. Convert any resulting `MatMulNBits` nodes to signed INT4 QDQ with the explicit `MatMulNBitsToQDQ` pass.

The vision encoder is not exported.
