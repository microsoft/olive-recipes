# Phi-4 Reasoning Plus Model Optimization

This repository demonstrates the optimization of the [Microsoft Phi-4 Reasoning Plus](https://huggingface.co/microsoft/Phi-4-reasoning-plus) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these main workflows:

- OpenVINO for Intel NPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`
