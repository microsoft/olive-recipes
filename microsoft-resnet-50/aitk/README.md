# ResNet optimization

This folder contains examples of ResNet optimization using different workflows.

- QDQ for Qualcomm NPU / AMD NPU
- OpenVINO for Intel NPU

## QDQ for Qualcomm NPU / AMD NPU

This workflow performs ResNet optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

## Evaluation result

The quantization uses 256 samples from train split of imagenet-1k dataset and the evaluations uses 256 samples from test split of imagenet-1k dataset.

| Activation Type&nbsp; | Weight Type&nbsp; | Size&nbsp; | Accuracy&nbsp; | Latency (avg)&nbsp; |
| --------------------- | ----------------- | ---------- | -------------- | ------------------- |
| float32               | float32           | 97.3 MB    | -              | -                   |
| QUInt16               | QUInt8            | 24.5MB     | 0.78515625     | 2.53724 ms          |
