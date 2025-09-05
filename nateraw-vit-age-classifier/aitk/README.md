# Vit optimization

This folder contains examples of Vit optimization using different workflows.

- QDQ for AMD NPU
- OpenVINO for Intel NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

## Pipeline

It performs the optimization pipeline:

- *HF Model -> Onnx Model -> Quantized Onnx Model*

## Hardware Test Results

| Platform | Accuracy (%) | Latency Avg (ms) |
|----------|--------------|------------------|
| **Qualcomm NPU**<br/>Snapdragon X 12-core<br/>16.0 GB RAM | 63.83 | 41.82 |
| **AMD NPU**<br/>Ryzen AI 9 H 365<br/>32.0 GB RAM | 62.5 | 15.46 |
| **Intel NPU**<br/>Core Ultra 5 228V<br/>32.0 GB RAM | 62.94 | 10.71 |
| **DML**<br/>AMD Ryzen 9 7845HX with Radeon Graphics <br/>32.0 GB RAM  | 63.83 | 4.80 |
| **NVIDIA**<br/>AMD Ryzen 9 7845HX with Radeon Graphics <br/>NVIDIA GeForce RTX 4060<br/>32.0 GB RAM | 63.83 | 3.02 |