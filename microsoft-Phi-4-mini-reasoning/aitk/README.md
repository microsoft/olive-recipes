# Phi-4 Model Optimization

This repository demonstrates the optimization of the [Microsoft Phi-4 Mini Reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these main workflows:

- OpenVINO for Intel NPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`
