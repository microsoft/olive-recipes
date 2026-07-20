# Qwen3-0.6B Model Optimization

This repository demonstrates the optimization of the [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model. The optimization process is divided into these workflows:

- PTQ + AOT for QNN NPU

## PTQ + AOT for QNN NPU

This workflow contains two steps:

### Post-training Quantization (PTQ)

This step using GPTQModel library, MatMul-NBits-QDQ and Static Quantization etc. They are resource-intensive and requires GPU acceleration.

### Ahead of Time (AOT) Compilation

This step compiles model using QNN Execution Provider in a separate Python environment with onnxruntime-qnn installed. Note that after compilation, the model must run on an EP with same or higher version of QAIRT SDK as the package (https://github.com/onnxruntime/onnxruntime-qnn/releases#release-v2.1.0).
