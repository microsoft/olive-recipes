# Qwen2.5-Coder-0.5B-Instruct Model Optimization

This repository demonstrates the optimization of the [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) model using **post-training quantization (PTQ)** techniques.

- OpenVINO for Intel® GPU/NPU

## Intel® Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### **Inference**

#### **Run Console-Based Chat Interface**
Execute the provided `inference_sample.ipynb` notebook.
