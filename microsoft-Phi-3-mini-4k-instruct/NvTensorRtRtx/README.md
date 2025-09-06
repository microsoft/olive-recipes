# Phi-3-mini-4k-instruct optimization

This folder contains examples of Olive recipes for `Phi-3-mini-4k-instruct` optimization.

## INT4 Model Building

The olive recipe ` Phi-3-mini-4k-instruct_model_builder_int4.json` uses `ModelBuilder` and `MatMulNBitsToQDQ` passes to generate the INT4 model for `NvTensorRTRTXExecutionProvider` (aka `NvTensorRtRtx` EP).

### Setup

1. Install Olive 

2. Install onnxruntime-genai package that has support for NvTensorRTRTXExecutionProvider.

### Steps to run

Use the following command to export the model using Olive with NvTensorRTRTXExecutionProvider:

```bash
olive run --config Phi-3-mini-4k-instruct_model_builder_int4.json
```