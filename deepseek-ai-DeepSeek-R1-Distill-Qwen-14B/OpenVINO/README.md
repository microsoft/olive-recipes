# DeepSeek-R1-Distill Quantization

This folder contains a sample use case of Olive to optimize a [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) model using OpenVINO tools.

- Intel® GPU: [DeepSeek-R1-Distill-Qwen-14B Dynamic shape model](#deepseek-r1-distill-qwen-14b-gpu-context-ov-dy-gs128-r1)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Dynamic shape model

The workflow in Config file: [DeepSeek-R1-Distill-Qwen-14B-gpu-context-ov-dy-gs128-r1.json](DeepSeek-R1-Distill-Qwen-14B-gpu-context-ov-dy-gs128-r1.json) executes the above workflow producing a dynamic shape model for GPU.

## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model for GPU

```bash
olive run --config DeepSeek-R1-Distill-Qwen-14B-gpu-context-ov-dy-gs128-r1.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("DeepSeek-R1-Distill-Qwen-14B-gpu-context-ov-dy-gs128-r1.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

The sample command to run after all setup would be as follows:-

```bash
python model-chat.py -e follow_config -v -g -m models/DeepSeek_R1_Distill_Qwen_14B_gpu_context_ov_dy_gs128_r1/model/
```