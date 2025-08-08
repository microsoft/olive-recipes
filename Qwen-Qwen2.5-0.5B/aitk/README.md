# Qwen2.5-0.5B Model Optimization

This repository demonstrates the optimization of the [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) model using **post-training quantization (PTQ)** techniques.

- OpenVINO for Intel GPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`

### **Inference**

#### **Run Console-Based Chat Interface**

Execute the provided `inference_sample.ipynb` notebook.
