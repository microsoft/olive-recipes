# Qwen3.6-35B-A3B optimization

This folder contains an Olive recipe for exporting the text-only component of `Qwen/Qwen3.6-35B-A3B` for the
`NvTensorRTRTXExecutionProvider` (also known as the `NvTensorRtRtx` EP).

## INT4 weight-only quantization

The `Qwen3.6-35B-A3B_model_builder_int4.json` recipe uses the ONNX Runtime GenAI Qwen3.5 MoE hybrid model builder,
which matches the architecture declared by the Qwen3.6 checkpoint, to:

1. Export a standalone text model by including the token embedding layer (`exclude_embeds=false`).
2. Apply symmetric INT4 RTN weight-only quantization with a block size of 32.
3. Convert any resulting `MatMulNBits` nodes to signed INT4 QDQ with the explicit `MatMulNBitsToQDQ` pass.
4. Enable the shared past/present buffer and CUDA graph capture for TRT-RTX inference.
5. Repair affected ONNX Runtime GenAI exports so every MoE layer computes
   `shared_expert_output * sigmoid(shared_expert_gate)` before combining the shared and routed experts.

The vision encoder is not exported.

The `export.py` entry point runs Olive, applies the shared-expert correction when needed, and validates that the final
model contains signed INT4 QDQ weights, no `MatMulNBits` nodes, one shared-expert gate per `QMoE` layer, a shared
past/present buffer, and CUDA graph configuration. The command fails instead of silently returning an incompatible
model if any requirement is missing.

## Setup

1. Install Olive.
2. Install a Transformers 5.x release that recognizes the `qwen3_5_moe` architecture.
3. Install an ONNX Runtime GenAI package or build that supports Qwen3.5 MoE hybrid models and the
   `NvTensorRTRTXExecutionProvider`.

## Run

```bash
python export.py -o output
```

The TRT-RTX-ready model is written to `output/model.onnx`.
