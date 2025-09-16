# BERT optimization

This folder contains examples of BERT optimization using different workflows.

- QDQ for Qualcomm NPU / AMD NPU
- OpenVINO for Intel® CPU/GPU/NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

## QDQ for Qualcomm NPU / AMD NPU

This workflow performs BERT optimization on NPU with ONNX Runtime QDQ. It performs the optimization pipeline:

- PyTorch Model -> Onnx Model -> Static shaped Onnx Model -> Quantized Onnx Model

## Intel® Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

## Results

On a Snapdragon(R) X 12-core X1E80100:

|Model|Runtime|Size|Accuracy|Latency (ms)|
|-|-|-|-|-|
|Conversion Only|CPU|417 MB|0.90|50.41|
|Quantized|Qualcomm NPU|126 MB|0.90|35.50|
