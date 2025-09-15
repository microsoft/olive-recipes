# Optimize meta-llama/Llama-2-7b-hf

## Install dependencies

Run the following command:

```bash
python -m pip install -r requirements.txt
```

## Optimize meta-llama/Llama-2-7b-hf using Olive optimize CLI

Run the following command to optimize the model for CUDAExecutionProvider:

```bash
olive optimize -m meta-llama/Llama-2-7b-hf --device gpu --provider CUDAExecutionProvider --log_level 1 --precision int4
```

The optimized ONNX model is saved in `optimized-model` folder.

## (Optional) Running ONNX Model on CPU or GPU in a Console-Based Chat Interface

To run the ONNX GenAI model, please set up the latest ONNXRuntime GenAI.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.
