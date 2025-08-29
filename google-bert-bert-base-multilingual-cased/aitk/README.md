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
| QNN                   | 12.46                | 151.80                       | facebook/xnli |
| Intel NPU             | 4.54                 |                              | wikipedia     |
| Intel GPU             | 2.85                 |                              | wikipedia     |
| Intel CPU             | 4.30                 |                              | wikipedia     |
| AMD NPU               | 11.95                | 83.54                        | facebook/xnli |
| NVIDIA TRT            | 1.95                 | 492.08                       | facebook/xnli |
| DirectML              | 6.05                 | 179.68                       | facebook/xnli |
|-----------------------|----------------------|------------------------------|---------------|

*Note: Latency can vary significantly depending on the hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*
