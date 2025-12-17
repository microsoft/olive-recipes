# Phi 3 mini instruct Quantization

This folder contains a sample use case of Olive to optimize a Phi-3-mini-instruct models using OpenVINO tools.

- Intel® GPU: [Phi 3 mini 128k Instruct Dynamic Shape Model](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- NVModelOptQuantization for NVIDIA TRT for RTX GPU

## Intel® Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Phi 3 Dynamic shape model

The following config files executes the above workflow producing as dynamic shaped model:

1. [phi3_ov_config.json](phi3_ov_config.json)

## NVModelOptQuantization for NVIDIA TRT for RTX GPU

To run this workflow, you need to [install CUDA](https://developer.nvidia.com/cuda-toolkit-archive) as required in [Doc](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/windows/_installation_for_Windows.html).
