# Llama3.1 8B Quantization

This folder contains a sample use case of Olive to optimize a [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model using OpenVINO tools.

- Intel® GPU: [Llama-3.1 8B Instruct GPU](#llama3-1-8b-ov-gpu-config)
- Intel® NPU: [Llama-3.1 8B Instruct NPU](#llama3-1-8b-ov-npu-config)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Llama3-1 8B OV GPU Config
The workflow in Config file: [llama3_1_8b_ov_gpu_config.json](llama3_1_8b_ov_gpu_config.json) executes the above workflow producing a dynamic shape model.

### Llama3-1 8B OV NPU Config
The workflow in Config file: [llama3_1_8b_ov_npu_config.json](llama3_1_8b_ov_npu_config.json) executes the above workflow producing a dynamic shape model.

## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install -r requirements.txt
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model

```bash
olive run --config llama3_1_8b_ov_gpu_config.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("llama3_1_8b_ov_gpu_config.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) GitHub repository.

The sample command to run after all setup would be as follows:-

```bash
python model-chat.py -e follow_config -v -g -m model/Llama-3.1-8B-Instruct-ov-gpu-int4/
```
