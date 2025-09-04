# Export Gemma Model to ONNX

## google/gemma-3-270m-it

### Install dependencies

Install latest `olive-ai` and `onnxruntime-genai` nightly package:

```bash
python -m pip install olive-ai - U
python -m pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-genai
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
