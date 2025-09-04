# Qwen 2.5 Coder Quantization

This folder contains a sample use case of Olive to optimize a Qwen 2.5 Coder models using OpenVINO tools.

- Intel® GPU: [Qwen 2.5 Coder 3B Dynamic Shape Model](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Dynamic shape model sym bkp int8 sym r1

The flow in the following config files executes the above workflow producing a dynamic shape model.

1. [Qwen2.5-Coder-3B-Instruct-gpu-context-dy.json](Qwen2.5-Coder-3B-Instruct-gpu-context-dy.json)


## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model using the following command:

```bash
olive run --config <config_file.json>
```

Example:
```bash
olive run --config Qwen2.5-Coder-3B-Instruct-gpu-context-dy.json
```
or run simply with python code:

```python
from olive import run
workflow_output = run("Qwen2.5-Coder-3B-Instruct-gpu-context-dy.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

The sample command to run after all setup would be as follows:

```bash
python model-chat.py -e follow_config -v -g -m models/<model_folder>/model/
```

Example:

```bash
python model-chat.py -e follow_config -v -g -m models/Qwen2.5_Coder_3B_Instruct_gpu_context_dy/model/
```