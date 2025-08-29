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
| QNN                   | 11.17                | 58.51                        | facebook/xnli |
| Intel NPU             | 4.80                 |                              | wikipedia     |
| Intel GPU             | 3.00                 |                              | wikipedia     |
| Intel CPU             | 4.80                 |                              | wikipedia     |
| AMD NPU               | 11.98                | 87.37                        | facebook/xnli |
| NVIDIA TRT            | 2.34                 | 507.45                       | facebook/xnli |
| DirectML              | 13.73                | 149.38                       | facebook/xnli |
|-----------------------|----------------------|------------------------------|---------------|

*Note: Latency can vary significantly depending on the hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*
