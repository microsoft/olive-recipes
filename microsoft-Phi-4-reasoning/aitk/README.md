# Phi-4 Reasoning Model Optimization

This repository demonstrates the optimization of the [Microsoft Phi-4 Reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these main workflows:

- OpenVINO for Intel NPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`
