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
| **Qualcomm NPU**<br/>Snapdragon X 12-core<br/>16.0 GB RAM | 75 | 7.93 |
| **AMD NPU**<br/>Ryzen AI 9 H 365<br/>32.0 GB RAM | 75.78 | 28.24 |
| **Intel NPU**<br/>Core Ultra 5 228V<br/>32.0 GB RAM | 73.10 | 5.36 |
| **DML**<br/>AMD Ryzen 9 7845HX with Radeon Graphics <br/>32.0 GB RAM  | 76.33 | 1.34 |
| **NVIDIA**<br/>AMD Ryzen 9 7845HX with Radeon Graphics <br/>NVIDIA GeForce RTX 4060<br/>32.0 GB RAM | 76.33 | 4.36 |