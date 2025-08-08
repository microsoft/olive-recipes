# Openai Clip optimization

This folder contains examples of Openai Clip optimization using different workflows.

- Text and vision model QDQ for Qualcomm NPU
- QDQ for AMD NPU
- OpenVINO for Intel NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

## Openai Clip text optimization with QDQ for Qualcomm NPU

This example performs Openai Clip optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

### Evaluation result

The quantization uses 12 samples from Flickr30k dataset and the evaluations uses 100 samples from Flickr30k dataset.

| Activation Type | Weight Type | Size | Latency ms (avg) | Accuracy (%) | Mean Similarity |
| --------------------- | ----------------- | ---------- | ---------------------- | ------------ | --------------- |
| QUInt16               | QUInt8            | 100        | 28.79                  | 100.0        | 0.9881          |

## Openai Clip vision optimization with QDQ for Qualcomm NPU

This example performs Openai Clip optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

### Evaluation result

The quantization uses 100 samples from mini-ImageNet dataset and the evaluations uses 100 samples from mini-ImageNet dataset.

| Activation Type | Weight Type | Size | Latency ms (avg) | Accuracy (%) | Mean Similarity |
| --------------------- | ----------------- | ---------- | ---------------------- | ------------ | --------------- |
| QUInt16               | QUInt8            | 100        | 267.73                 | 100.0        | 0.9975          |

## Openai Clip optimization with QDQ for AMD NPU

This example performs Openai Clip optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

### Evaluation result

The quantization uses 100 samples from Flickr30k dataset and the evaluations uses 100 samples from Flickr30k dataset.

| Activation Type | Weight Type | Latency ms (avg) | Throughput (samples/sec) | Accuracy (%) |
| --------------------- | ----------------- | ---------------------- | ----------------------- | ------------ |
| QUInt8               | QUInt8            | 215.54                 | 4.51                    | 100.0        |

## Openai Clip optimization with OpenVINO

This example performs Openai Clip optimization with OpenVINO in one workflow for Intel NPU.

### Evaluation result

The quantization uses conceptual captions dataset and the evaluations uses 10 samples from Flickr30k dataset.

| Latency ms (avg) | Throughput (samples/sec) | Accuracy (%) |
| ---------------------- | ----------------------- | ------------ |
| 71.24                  | 13.88                   | 100.0        |

## Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Float16 Onnx Model*

## Hardware Test Results

| Platform | Model | Accuracy (%) | Degrad (%) | Mean Similarity | Latency Avg (ms) | Latency P90 (ms) | Latency Max (ms) | Latency Min (ms) | Throughput Avg (samples/sec) | Throughput Max (samples/sec) | Throughput Min (samples/sec) |
|----------|-------|--------------|------------|-----------------|------------------|------------------|------------------|------------------|------------------------------|------------------------------|------------------------------|
| **Qualcomm NPU**<br/>Snapdragon X 12-core<br/>16.0 GB RAM | Text Model | 100.0 | 1.19 | 0.9881 | 28.79 | 50.13 | - | - | - | - | - |
| | Vision Model | 100.0 | 0.25 | 0.9975 | 267.73 | 286.72 | - | - | - | - | - |
| **AMD NPU**<br/>Ryzen AI 9 H 365<br/>32.0 GB RAM | CLIP Model | 100.0 | - | - | 215.54 | - | 238.72 | 189.61 | 4.51 | 5.08 | 4.10 |
| **Intel NPU**<br/>Core Ultra 5 228V<br/>32.0 GB RAM | CLIP Model | 100.0 | - | - | 71.24 | 72.38 | - | - | 13.88 | 14.10 | 13.52 |