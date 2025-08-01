# DeepSeek-R1-Distill-Qwen-14B Model Optimization

This repository demonstrates the optimization of the [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) model using **post-training quantization (PTQ)** techniques. 
- OpenVINO for Intel GPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`

#### **Run Console-Based Chat Interface**
Execute the provided `inference_sample.ipynb` notebook.
