# Mistral-7B-Instruct-v0.3 Optimization

This repository demonstrates the optimization of the [Mistral 7B Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model. 

The optimization process is divided into these main workflows:
- OpenVINO for Intel GPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`
