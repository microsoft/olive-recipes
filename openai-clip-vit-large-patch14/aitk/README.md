# Openai Clip optimization

This folder contains examples of Openai Clip optimization using different workflows.

- QDQ for Qualcomm NPU
- QDQ for AMD NPU
- Float16 for AMD GPU (MIGraphX)
- OpenVINO for Intel NPU
- Float16 for NVIDIA TRT RTX GPU
- Float16 for DML (general GPU)

## Openai Clip optimization with QDQ for Qualcomm NPU

This example performs Openai Clip optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

Uses `QNNExecutionProvider` for Qualcomm NPU.

## Openai Clip optimization with QDQ for AMD NPU

This example performs Openai Clip optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Transformer Optimization -> Quantized Onnx Model*

Uses `VitisAIExecutionProvider` for AMD NPU.

## Openai Clip optimization with Float16 for AMD GPU

This example performs Openai Clip optimization with float16 downcasting for AMD GPU. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Float16 Onnx Model*

Uses `MIGraphXExecutionProvider` for AMD GPU.

## Openai Clip optimization with OpenVINO

This example performs Openai Clip optimization with OpenVINO in one workflow for Intel NPU. It performs the optimization pipeline:

- *PyTorch Model -> OpenVINO IR -> Quantized OpenVINO IR*

Uses `OpenVINOExecutionProvider` for Intel NPU.

## Float16 optimization for NVIDIA TRT RTX GPU

This example performs Openai Clip optimization with float16 downcasting. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Float16 Onnx Model*

Uses `NvTensorRTRTXExecutionProvider` for NVIDIA RTX GPU.

## Float16 optimization for DML (general GPU)

This example performs Openai Clip optimization with float16 downcasting. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Float16 Onnx Model*

Uses `DmlExecutionProvider` for general GPU support via DirectML.

## Hardware Test Results

| Platform | Dataset | Accuracy (%)  | Latency Avg (ms) | Latency P90 (ms) | Latency Max (ms) | Latency Min (ms) | Throughput Avg (samples/sec) | Throughput Max (samples/sec) | Throughput Min (samples/sec) |
|----------|---------|--------------|------------------|------------------|------------------|------------------|------------------------------|------------------------------|------------------------------|
| **Qualcomm NPU**<br/>Snapdragon X 12-core<br/>16.0 GB RAM  | Flickr30k (100 samples) | 99.0 | 386.68 | - | 393.46 | 380.80 | 2.50 | 2.66 | 1.43 |
| **AMD NPU**<br/>Ryzen AI 9 H 365<br/>32.0 GB RAM  | Flickr30k (10 samples) | 100.0 |  215.54 | - | 238.72 | 189.61 | 4.51 | 5.08 | 4.10 |
| **Intel NPU**<br/>Core Ultra 5 228V<br/>32.0 GB RAM  | Flickr30k (10 samples) | 100.0 | 71.24 | 72.38 | - | - | 13.88 | 14.10 | 13.52 |
| **DML**<br/>Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz (3.70 GHz)<br/>32.0 GB RAM   | Flickr30k (10 samples) | 100.0 | 11.37 | - | 12.09 | 10.99 | 87.62 | 91.38 | 81.50 |
| **NVIDIA**<br/>Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz (3.70 GHz) <br/>NVIDIA GeForce RTX 4080<br/>32.0 GB RAM  | Flickr30k (10 samples) | 100.0 | 10.46 | - | 20.33 | 8.98 | 107.77 | 119.52 | 87.15 |
