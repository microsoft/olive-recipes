# DeepSeek-R1-Distill-Qwen-14B Model Optimization

This repository demonstrates the optimization of the [DeepSeek R1 Distill Qwen 14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these workflows:

- Intel® GPU: [DeepSeek R1 Distill Qwen 14B Dynamic Shape Model](./deepseek_ov_config.json)

## Intel® Workflows

These workflows performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*
