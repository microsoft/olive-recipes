# Google BERT optimization

This folder contains examples of Google BERT (bert-large-uncased) optimization using different workflows.

- QDQ for AMD NPU
- QDQ for Qualcomm NPU
- OpenVINO for Intel NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

## Google BERT optimization with QDQ for AMD NPU

This example performs Google BERT optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

### Evaluation result

The quantization uses 10 samples from XNLI (English) validation dataset and the evaluations uses 10 samples from XNLI (English) validation dataset.

| Activation Type | Weight Type | Latency ms (avg) | Throughput (samples/sec) | 
| --------------- | ----------- | ---------------- | ----------------------- | 
| QUInt16         | QUInt8      | TBD              | TBD                     | 

## Google BERT optimization with QDQ for Qualcomm NPU

This example performs Google BERT optimization with QDQ in one workflow. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

### Evaluation result

The quantization uses 10 samples from XNLI (English) validation dataset and the evaluations uses 10 samples from XNLI (English) validation dataset.

| Activation Type | Weight Type | Size | Latency ms (avg) | Mean Similarity |
| --------------- | ----------- | ---- | ---------------- | --------------- |
| QUInt16         | QUInt8      | 10   | TBD              | TBD             |

## Google BERT optimization with OpenVINO

This example performs Google BERT optimization with OpenVINO in one workflow for Intel NPU.

### Evaluation result

The quantization uses XNLI dataset and the evaluations uses 10 samples from XNLI (English) validation dataset.

| Latency ms (avg) | Throughput (samples/sec) |
| ---------------- | ----------------------- |
| TBD              | TBD                     |

## Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Float16 Onnx Model*

## Hardware Test Results

| Platform | Model | Latency Avg (ms) | Latency P90 (ms) | Latency Max (ms) | Latency Min (ms) | Throughput Avg (samples/sec) | Throughput Max (samples/sec) | Throughput Min (samples/sec) |
|----------|-------|------------------|------------------|------------------|------------------|------------------------------|------------------------------|------------------------------|
| **AMD NPU**<br/>Ryzen AI 9 H 365<br/>32.0 GB RAM | BERT-large | TBD | - | TBD | TBD | TBD | TBD | TBD |
| **Qualcomm NPU**<br/>Snapdragon X 12-core<br/>16.0 GB RAM | BERT-large | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Intel NPU**<br/>Core Ultra 5 228V<br/>32.0 GB RAM | BERT-large | TBD | TBD | - | - | TBD | TBD | TBD |
| **DML**<br/>Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz (3.70 GHz)<br/>32.0 GB RAM | BERT-large | TBD | - | TBD | TBD | TBD | TBD | TBD |
| **NVIDIA**<br/>Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz (3.70 GHz)<br/>NVIDIA GeForce RTX 4080<br/>32.0 GB RAM | BERT-large | TBD | - | TBD | TBD | TBD | TBD | TBD |

## Notes

- The model used is `google-bert/bert-large-uncased` for feature extraction tasks
- Evaluation metrics focus on latency and throughput for feature extraction performance
- TBD values will be populated after running the optimization workflows and benchmarks