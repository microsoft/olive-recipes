# Gemma-3-1b-it Compression

This folder contains a sample use case of Olive to optimize the [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) model using Intel® OpenVINO tools.

- Intel® NPU: [Gemma 3 1b it Dynamic Shape model optimized for NPU](#gemma-3-1b-it-npu)
- Intel® GPU: [Gemma 3 1b it Dynamic Shape model optimized for GPU](#gemma-3-1b-it-gpu)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *Huggingface Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Gemma 3 1b it NPU

The flow in the following config file executes the above workflow producing a dynamic shape model.

1. [gemma_3_1b_it_context_ov_npu_config.json](gemma_3_1b_it_context_ov_npu_config.json)

### Gemma 3 1b it GPU

The flow in the following config file executes the above workflow producing a dynamic shape model.

1. [gemma_3_1b_it_context_ov_gpu_config.json](gemma_3_1b_it_context_ov_gpu_config.json)

## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
python -m pip install -r requirements.txt
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model using the following command:

```bash
olive run --config <config_file.json>
```

Example:

```bash
olive run --config gemma_3_1b_it_context_ov_npu_config.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("<config_file.json>")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) GitHub repository.

```bash
python model-chat.py -e follow_config -v -g -m models/<model_folder>/
```

Example:

```bash
python model-chat.py -e follow_config -v -g -m models/gemma_3_1b_it_context_ov_npu/
```
