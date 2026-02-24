# Qwen2.5-VL-3B-Instruct with onnx-genai-models

Export and run Qwen2.5-VL-3B-Instruct using [onnx-genai-models](https://github.com/onnx/onnx-genai-models) and [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai).

## Setup

```bash
conda activate new_mb
pip install onnx-genai-models onnxruntime-genai
```

## Export

```bash
python optimize.py
```

This builds a 3-model split required by onnxruntime-genai:

| File | Description |
|------|-------------|
| `vision.onnx` | Vision encoder (ViT with packed attention + spatial merge) |
| `embedding.onnx` | Token embedding + vision/text mixer (ScatterND) |
| `decoder.onnx` | Text decoder with GroupQueryAttention + MRoPE |
| `decoder.onnx.data` | External weight data for decoder |
| `genai_config.json` | Model metadata for onnxruntime-genai |
| `processor_config.json` | Image preprocessing pipeline config |
| `tokenizer.json` | Tokenizer |

Options:

```bash
python optimize.py --dtype f16        # float16 (default)
python optimize.py --dtype f32        # float32
python optimize.py --output my_model  # custom output directory
python optimize.py --no-weights       # graph-only export (no weights)
```

## Inference

```bash
# Text-only
python inference.py --prompt "What is the capital of France?"

# With image
python inference.py --prompt "Describe this image" --image cat.jpeg

# Interactive mode
python inference.py --interactive
```
