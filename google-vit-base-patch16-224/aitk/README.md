# Vision Transformer (ViT) Optimization

This folder contains examples of VIT optimization using different workflows.

- QDQ for Qualcomm NPU / AMD NPU
- OpenVINO for Intel NPU

## Optimization Workflows

### ViT optimization with qdq

This example performs ViT optimization in one workflow. It performs the optimization pipeline:

- *Huggingface Model -> Onnx Model -> Quantized Onnx Model*
