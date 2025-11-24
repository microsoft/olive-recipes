# Phi 3 mini instruct Quantization

This folder contains a sample use case of Olive to optimize a Phi-3-mini-instruct models using OpenVINO tools.

- Intel® GPU: [Phi 3 Mini 4k Instruct Dynamic Shape Model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- Intel® NPU: [Phi 3 Mini 4k Instruct Dynamic Shape Model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- ModelBuilder for NVIDIA TRT for RTX GPU

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Phi 3 Dynamic shape model

The following config files executes the above workflow producing as dynamic shaped model:

1. [phi3_ov_config.json](phi3_ov_config.json)
2. [phi3_ov_npu_config.json](phi3_ov_npu_config.json)

## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model:

```bash
olive run --config <config_file.json>
```

Example:

```bash
olive run --config phi3_ov_config.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("<config_file.json>")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
