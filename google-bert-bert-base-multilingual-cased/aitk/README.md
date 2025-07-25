# BERT Optimization

This folder contains examples of BERT optimization using different workflows.

- QDQ for Qualcomm NPU / AMD NPU
- OpenVINO for Intel NPU

## BERT Quantization QDQ

This workflow quantizes the model. It performs the pipeline:
- *HF Model-> ONNX Model ->Quantized Onnx Model*

Config file: `bert-base-multilingual-cased_qdq.json`

### Latency / Throughput

| Model Version         | Latency (ms/sample)  | Throughput (token per second)| Dataset       |
|-----------------------|----------------------|------------------------------|---------------|
| PyTorch FP32          | 1162                 | 0.81                         | facebook/xnli |
| ONNX INT8 (QDQ)       | 590                  | 1.75                         | facebook/xnli |

*Note: Latency can vary significantly depending on the hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*
