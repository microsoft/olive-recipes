# DeepSeek-R1-Distill-Qwen-1.5B optimization

This folder contains examples of Olive recipes for `DeepSeek-R1-Distill-Qwen-1.5B` optimization.

## FP16 Model Building

The olive recipe `DeepSeek-R1-Distill-Qwen-1.5B_fp16_model_builder.json` uses `ModelBuilder` pass to generate the FP16 model for `NvTensorRTRTXExecutionProvider` (aka `NvTensorRtRtx` EP).

### Setup

1. Install Olive 

2. Install onnxruntime-genai package that has support for NvTensorRTRTXExecutionProvider.

### Steps to run

Use the following command to export the model using Olive with NvTensorRTRTXExecutionProvider:

```bash
olive run --config DeepSeek-R1-Distill-Qwen-1.5B_fp16_model_builder.json
```