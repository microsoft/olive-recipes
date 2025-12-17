# Laion Clip optimization

This folder contains examples of Laion Clip optimization using different workflows.

- QDQ for Qualcomm NPU / AMD NPU
- OpenVINO for Intel® CPU/GPU/NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

## Laion Clip optimization with QDQ for Qualcomm NPU / AMD NPU

This example performs Laion Clip optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

## Laion Clip optimization with OpenVINO

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

## Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Float16 Onnx Model*
