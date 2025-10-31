# Openai Clip optimization

This folder contains examples of Openai Clip optimization using different workflows.

- OpenVINO for Intel® CPU/GPU/NPU

## Openai Clip optimization with OpenVINO

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*
