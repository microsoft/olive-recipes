# Export GPT-OSS-20b Model to ONNX

### Install dependencies

Install latest `olive-ai` and `onnxruntime-genai-cuda` nightly package:

```bash
python -m pip install -r requirements.txt
```

### Capture ONNX Graph Using Olive CLI

Run the following command:

```bash
olive capture-onnx-graph                                        \
  --model_name_or_path openai/gpt-oss-20b                       \
  --trust_remote_code                                           \
  --conversion_device gpu                                       \
  --use_model_builder                                           \
  --use_ort_genai                                               \
  --extra_mb_options int4_op_types_to_quantize=MatMul/Gather    \
  -o int4_cuda_int4_qmoe
```

The exported ONNX model is saved in `int4_cuda_int4_qmoe` folder.

### (Optional) Running ONNX Model on CPU or GPU in a Console-Based Chat Interface

To run the ONNX GenAI model, please set up the latest ONNXRuntime GenAI.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

```bash
python model-chat.py -m int4_cuda_int4_qmoe/model -e cuda
```
