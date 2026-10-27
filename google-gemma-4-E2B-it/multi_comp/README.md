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
2. Capture the quantized ONNX graphs with Mobius using `olive capture-onnx-graph`.
3. Run `ov.json` or `qnn.json`.

To run the steps manually with the currently configured accuracy level of `4`:

```bash
# 1. Quantize
olive run --config gemma4_quantize.json

# 2. Capture ONNX graphs
olive capture-onnx-graph \
    --model_name_or_path gemma4_quantized_hf \
    --use_mobius_builder \
    --precision fp32 \
    --matmul_nbits_accuracy_level 4 \
    --output_path gemma4_onnx

# 3. Optimize for one target
olive run --config ov.json
# Or:
olive run --config qnn.json
```

## Capture accuracy

Set `input_model.config.model_attributes.matmul_nbits_accuracy_level` in the
existing target recipe: `qnn.json` for QNN or `ov.json` for OpenVINO. Both are
configured with `4`:

```json
"model_attributes": {
    "matmul_nbits_accuracy_level": 4
}
```

`optimize.py` reads the selected recipe before quantization and forwards its value
as `--matmul_nbits_accuracy_level` to the same `olive capture-onnx-graph` command
with `--use_mobius_builder`. No separate export JSON is used. If running the
steps manually, use the selected recipe's value in the capture command above.

This overrides the activation-computation accuracy level
for all exported `MatMulNBits` operators, including the CPU embedding projection
and LM head retained after QNN conversion. It does not change weight bit widths
or `GatherBlockQuantized`, and is not a lossless numerical optimization.

The setting accepts integers `0` through `4`. Remove the setting or set it to
`null` to omit the capture argument and preserve Mobius's existing EP-specific
defaults; explicit `0` is forwarded and overrides any nonzero EP default.
Invalid values fail before quantization starts. This requires Olive and Mobius
versions that support the capture argument.

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
