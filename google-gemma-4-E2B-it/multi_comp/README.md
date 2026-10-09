# Gemma 4 E2B - Multi-component Optimization

This recipe optimizes for
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it):

## Prerequisites

```bash
pip install "git+https://github.com/microsoft/Olive.git"
pip install "git+https://github.com/onnxruntime/mobius.git"
pip install transformers torch onnxruntime-gpu onnxruntime-genai datasets pandas pyarrow pillow tokenizers requests
hf auth login
```

Quantization requires a CUDA GPU.

## Optimize

Run from this `multi_comp` directory, choosing one target:

```bash
# OpenVINO
python optimize.py --ep ov

# QNN
python optimize.py --ep qnn
```

For either target, the script runs exactly three steps:

1. Quantize the Hugging Face components with `gemma4_quantize.json`.
2. Capture the quantized ONNX graphs with Mobius.
3. Run `ov.json` or `qnn.json`.

To run the steps manually:

```bash
# 1. Quantize
olive run --config gemma4_quantize.json

# 2. Capture ONNX graphs
olive capture-onnx-graph \
    --model_name_or_path gemma4_quantized_hf \
    --use_mobius_builder \
    --precision fp32 \
    --output_path gemma4_onnx

# 3. Optimize for one target
olive run --config ov.json
# Or:
olive run --config qnn.json
```

## QNN

Install the QNN requirements in the QNN target Python environment:

```bash
pip install -r requirements-qnn.txt
```

## Inference

The examples below use the QNN output. Use `gemma4_ov` for OpenVINO.

Text:

```bash
python ../inference.py --model-path gemma4_qnn --prompt "What is the capital of France?" --verbose
```

Image:

```bash
python ../inference.py --model-path gemma4_qnn --image ../cat.jpeg --prompt "What animal is shown? Answer in one short sentence." --verbose
```
