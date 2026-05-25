# gpt-oss-20b Model Optimization

This recipe optimizes the [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) model for AMD NPU via the **VitisAI Execution Provider**.

The input is a **pre-quantized ONNX** model published by ONNX Runtime ([onnxruntime/gpt-oss-20b-onnx](https://huggingface.co/onnxruntime/gpt-oss-20b-onnx)), and the Olive workflow only runs the VitisAI NPU model generation pass (`VitisGenerateModelLLM`) — no additional quantization step is needed.

## Convert for AMD NPU

### 1. Download the pre-quantized ONNX model

```bash
hf download onnxruntime/gpt-oss-20b-onnx --include "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*" --local-dir ./models/gpt-oss-20b-onnx
```

### 2. Run the Olive recipe

```bash
olive run --config gpt-oss-20b_vitis_ai_config.json
```

Optimized model is saved to `model/gpt-oss-20b-vai/`.

## Inference

Run `inference_sample.ipynb` after the conversion finishes. Inference requires a Windows machine with an AMD NPU and the VitisAI Execution Provider available.
