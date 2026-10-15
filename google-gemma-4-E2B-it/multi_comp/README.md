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

## OpenVINO

Install OpenVINO in the optimization environment:

```bash
pip install openvino
```

The decoder pipeline keeps the Mobius ONNX export, applies QDQ conversion and
graph surgeries, then runs `OpenVINOConversion` and `OpenVINOEncapsulation`.
Conversion preserves dynamic shapes and the existing weight precision.
Encapsulation produces an ONNX `EPContext` targeting NPU, with the OpenVINO
IR (`.xml`/`.bin`) alongside it. The vision pipeline fixes the patch count,
infers intermediate shapes, then splits the encoder from the pooler/projector;
the embedding pipeline is unchanged.

Encapsulation uses `update_genai_config: true` to preserve the Mobius GenAI
configuration and automatically merge the decoder filename and NPU session
options into `gemma4_ov/genai_config.json`. It also disables
`search.past_present_share_buffer` for the encapsulated decoder. This requires
an Olive build with existing-GenAI-config update support in
`OpenVINOEncapsulation`; no manual runtime configuration edits are needed.
Inference requires an ONNX Runtime build with the OpenVINO EP and the matching
Intel NPU runtime.

## QNN

Install the QNN requirements in the QNN target Python environment:

```bash
pip install -r requirements-qnn.txt
```

`qnn.json` excludes `lm_head` from MatMulNBits-to-QDQ conversion, preserving its
8-bit, group-size-128 `MatMulNBits`. The exclusion uses the exact exported node
name; update it if a new ONNX capture changes that name.

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
