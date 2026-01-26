# BERT Optimization

This folder contains examples of BERT optimization using different workflows.

- QDQ for Qualcomm NPU / AMD NPU
- OpenVINO for Intel® CPU/GPU/NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU / WebGPU for general GPU

## QDQ for Qualcomm NPU / AMD NPU

This workflow quantizes the model. It performs the pipeline:
- *HF Model-> ONNX Model ->Quantized Onnx Model*

## Intel® Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Latency / Throughput

| Model Version         | Latency (ms/sample)  | Throughput (token per second)| Dataset       |
|-----------------------|----------------------|------------------------------|---------------|
| PyTorch FP32          | 1162                 | 0.81                         | facebook/xnli |
| ONNX INT8 (QDQ)       | 590                  | 1.75                         | facebook/xnli |

*Note: Latency can vary significantly depending on the hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*
