# DeepSeek-R1-Distill-Llama-8B Model Optimization

This repository demonstrates the optimization of the [DeepSeek R1 Distill Llama 8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these workflows:

- Intel® NPU: [DeepSeek R1 Distill Llama 8B Dynamic Shape Model](./deepseek_ov_npu_config.json)
- Intel® GPU: [DeepSeek R1 Distill Llama 8B Dynamic Shape Model](./deepseek_ov_config.json)

## Intel® Workflows

These workflows performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*
