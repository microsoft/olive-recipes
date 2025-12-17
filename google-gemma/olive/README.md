# Export Gemma Model to ONNX

## google/gemma-3-270m-it

### Install dependencies

Run the following command:

```bash
python -m pip install -r requirements.txt
```

### Capture ONNX Graph Using Olive CLI

Run the following command:

```bash
python -m olive capture-onnx-graph -m google/gemma-3-270m-it --use_model_builder -o output_model
```

The exported ONNX model is saved in `output_model` folder.

### (Optional) Running ONNX Model on CPU or GPU in a Console-Based Chat Interface

To run the ONNX GenAI model, please set up the latest ONNXRuntime GenAI.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

## google/gemma-3-1b-it

```bash
python -m pip install -r requirements.txt
# Use the following command to export the model using Olive with CPUExecutionProvider at FP32 precision:
olive run --config gemma-3-1b-it_model_builder_cpu_fp32.json
```
