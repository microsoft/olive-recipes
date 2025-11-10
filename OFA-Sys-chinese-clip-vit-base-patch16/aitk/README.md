# Chinese CLIP optimization

This folder contains examples of Chinese CLIP optimization using different workflows.

- OpenVINO for Intel® CPU/GPU/NPU

## Chinese CLIP optimization with OpenVINO

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

## Metric

| EP | Device | Result |
|-|-|-|
| OpenVINO for NPU | Intel Core Ultra 7 258V | accuracy: 0.97<br>latency-avg: 79.89768<br>latency-p90: 83.91246 |
