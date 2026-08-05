# Llama-3.1-8B-Instruct optimization

This folder contains examples of Olive recipes for `Llama-3.1-8B-Instruct` optimization.

## INT4 Model Building

The olive recipe `Llama-3.1-8B-Instruct_model_builder_int4.json` uses `ModelBuilder` pass to generate the INT4 model for `NvTensorRTRTXExecutionProvider` (aka `NvTensorRtRtx` EP).

### Setup

1. Install Olive

2. Install onnxruntime-genai package that has support for NvTensorRTRTXExecutionProvider.

### Steps to run

Use the following command to export the model using Olive with NvTensorRTRTXExecutionProvider:

```bash
olive run --config Llama-3.1-8B-Instruct_model_builder_int4.json
```
