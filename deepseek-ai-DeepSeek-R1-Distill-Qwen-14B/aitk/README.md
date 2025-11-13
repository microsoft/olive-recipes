# DeepSeek-R1-Distill-Qwen-14B Model Optimization

This repository demonstrates the optimization of the [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these workflows:

- OpenVINO for Intel® GPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`
- NVModelOptQuantization for NVIDIA TRT for RTX GPU

## Intel® Workflows

These workflows performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

## NVModelOptQuantization for NVIDIA TRT for RTX GPU

To run this workflow, you need to [install CUDA](https://developer.nvidia.com/cuda-toolkit-archive) as required in [Doc](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/windows/_installation_for_Windows.html).
