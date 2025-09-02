# BERT Optimization

This folder contains examples of BERT optimization using different workflows.

- QDQ for Qualcomm NPU / AMD NPU
- OpenVINO for Intel NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

## QDQ for Qualcomm NPU / AMD NPU

This workflow quantizes the model. It performs the pipeline:
- *HF Model-> ONNX Model ->Quantized Onnx Model*

### Latency / Throughput

| EP                    | Latency (ms/sample)  | Throughput (token per second)| Dataset       |
|-----------------------|----------------------|------------------------------|---------------|
| QNN                   | 6.57                 | 183.00                       | facebook/xnli |
| Intel NPU             | 4.53                 |                              | wikipedia     |
| Intel GPU             | 2.96                 |                              | wikipedia     |
| Intel CPU             | 20.60                |                              | wikipedia     |
| AMD NPU               | 9.87                 | 105.61                       | facebook/xnli |
| NVIDIA TRT            | 1.98                 | 465.30                       | facebook/xnli |
| DirectML              | 7.00                 | 170.25                       | facebook/xnli |
|-----------------------|----------------------|------------------------------|---------------|

*Note: Latency can vary significantly depending on the hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*
