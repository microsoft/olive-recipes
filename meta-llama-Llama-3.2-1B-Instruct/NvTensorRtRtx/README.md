# Llama-3.2-1B-Instruct optimization

This folder contains examples of Olive recipes for `Llama-3.2-1B-Instruct` optimization.

## FP16 Model Building

The olive recipe `Llama-3.2-1B-Instruct_model_builder_fp16.json` uses `ModelBuilder` pass to generate the FP16 model for `NvTensorRTRTXExecutionProvider` (aka `NvTensorRtRtx` EP).

### Setup

1. Install Olive 

2. Install onnxruntime-genai package that has support for NvTensorRTRTXExecutionProvider.

### Steps to run

Use the following command to export the model using Olive with NvTensorRTRTXExecutionProvider:

```bash
olive run --config Llama-3.2-1B-Instruct_model_builder_fp16.json
```