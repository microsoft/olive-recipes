# Qwen2.5-0.5B Model Optimization

This repository demonstrates the optimization of the [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) model using **post-training quantization (PTQ)** techniques.

- OpenVINO for Intel® GPU

## Intel® Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

## Metrics

On a 13th Gen Intel(R) Core(TM) i7-1370P:

|Model|Runtime|Size|Throughtput|Latency (ms)|
|-|-|-|-|-|
|Optimized|Intel GPU|366 MB|35.97|27.80|

## **Inference**

### **Run Console-Based Chat Interface**

Execute the provided `inference_sample.ipynb` notebook.
