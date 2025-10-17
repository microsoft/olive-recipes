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
| **Qualcomm NPU**<br/>Snapdragon X 12-core<br/>16.0 GB RAM | 75 | 28.88 |
| **AMD NPU**<br/>Ryzen AI 9 H 365<br/>32.0 GB RAM | 75.39 | 16.13 |
| **Intel NPU**<br/>Core Ultra 5 228V<br/>32.0 GB RAM | 74.55 | 11.31 |
| **DML**<br/>AMD Ryzen 9 7845HX with Radeon Graphics <br/>32.0 GB RAM  | 75 | 3.04 |
| **NVIDIA**<br/>AMD Ryzen 9 7845HX with Radeon Graphics <br/>NVIDIA GeForce RTX 4060<br/>32.0 GB RAM | 75 | 1.90 |