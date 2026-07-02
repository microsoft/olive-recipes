# Gemma 4 E2B (google/gemma-4-E2B-it)

Olive recipes that export [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
to ONNX via the [`MobiusBuilder`](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
pass and (optionally) quantize the decoder with Olive's K-Quant (Q4_K_M)
pass for INT4 deployment.

Gemma 4 is an any-to-any multimodal model with vision, audio, and text
capabilities. The pipeline produces four ONNX components (decoder,
vision_encoder, audio_encoder, embedding) for use with ORT GenAI.
`MobiusBuilder` writes a fully-formed ORT GenAI package
(`genai_config.json`, `tokenizer.json`, `image_processor.json`,
`audio_feature_extraction.json`) alongside the ONNX files — no
post-processing required.

## Prerequisites

```bash
pip install olive-ai mobius-ai
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| CPU | `pip install onnxruntime-genai` |
| GPU (CUDA) | `pip install onnxruntime-genai-cuda` |
| OpenVINO (NPU / GPU) | `pip install onnxruntime-genai` + [OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) |

## Recipes

| Recipe | Pipeline | Output dir |
|---|---|---|
| `cpu/fp32/config.json` | `MobiusBuilder(fp32)` | `cpu/fp32/models` |
| `cpu/int4/config.json` | `MobiusBuilder(fp32)` → `OnnxKQuantQuantization(bits=4, block=32)` | `cpu/int4/models` |
| `cuda/fp16/config.json` | `MobiusBuilder(fp16)` | `cuda/fp16/models` |
| `cuda/int4/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block=32)` | `cuda/int4/models` |
| `openvino/npu/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block=32)` | `openvino/npu/models` |
| `openvino/gpu/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block=32)` | `openvino/gpu/models` |

K-Quant (Q4_K_M) is significantly faster with GPU acceleration —
install `cupy-cuda12x` for a 19–51× speedup during quantization.

The `openvino/*` recipes build the same portable INT4 ONNX and write a
`genai_config.json` whose `provider_options` select the OpenVINO EP with the
matching `device_type` (`NPU` for `openvino/npu`, `GPU` for `openvino/gpu`).
The OpenVINO EP compiles the graph for the target device at load time; ops it
does not support fall back to the CPU EP automatically. This requires the
OpenVINO EP support in `MobiusBuilder`
([microsoft/Olive](https://github.com/microsoft/Olive)) and the mobius
`openvino` EP ([onnxruntime/mobius](https://github.com/onnxruntime/mobius)).

## Build

```bash
# CPU, full precision
olive run --config cpu/fp32/config.json

# CPU, INT4 (K-Quant)
olive run --config cpu/int4/config.json

# CUDA, FP16
olive run --config cuda/fp16/config.json

# CUDA, INT4 (K-Quant)
olive run --config cuda/int4/config.json

# OpenVINO NPU, INT4 (K-Quant)
olive run --config openvino/npu/config.json

# OpenVINO GPU, INT4 (K-Quant)
olive run --config openvino/gpu/config.json
```

Each command produces the full ORT GenAI package in the recipe's
`output_dir`:

```
<output_dir>/
├── decoder/model.onnx          # Text decoder
├── vision_encoder/model.onnx   # Vision encoder
├── audio_encoder/model.onnx    # Audio encoder
├── embedding/model.onnx        # Embedding fusion
├── genai_config.json           # Runtime configuration
├── image_processor.json
├── audio_feature_extraction.json
├── tokenizer.json
└── tokenizer_config.json
```

## Inference

```bash
# Text-only (CPU, fp32)
python inference.py --prompt "What is the capital of France?"

# CPU INT4
python inference.py --variant int4 --prompt "Hello"

# CUDA INT4
python inference.py --device gpu --variant int4 --prompt "Explain quantum computing"

# Interactive mode
python inference.py --device gpu --variant int4 --interactive
```

## Evaluation

```bash
# MMLU Pro (default 100 samples), CPU
python eval.py

# CUDA INT4
python eval.py --device gpu --variant int4
```

## References

- Mobius docs: <https://github.com/onnxruntime/mobius>
- Olive `MobiusBuilder` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py>
- Olive `OnnxKQuantQuantization` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/kquant_quantization.py>
