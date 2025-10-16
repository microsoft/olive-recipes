# Vision Transformer (ViT) Optimization

This folder contains examples of VIT optimization using different workflows.

- QDQ for Qualcomm NPU / AMD NPU
- OpenVINO for Intel® CPU/GPU/NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

## Optimization Workflows

### QDQ for Qualcomm NPU / AMD NPU

This example performs ViT optimization in one workflow. It performs the optimization pipeline:

- *Huggingface Model -> Onnx Model -> Quantized Onnx Model*

### Intel® Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*
