# swin2SR-realworld-sr-x4-64-bsrgan-psnr Model Optimization

This repository demonstrates the optimization of the [swin2SR-realworld-sr-x4-64-bsrgan-psnr](https://huggingface.co/caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these workflows:

- OpenVINO for Intel® NPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`

## Intel® Workflows

These workflows performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*