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
2. Capture ONNX graphs with Olive's Mobius builder.
3. Run `ov.json` (which fixes the OpenVINO vision capacity) or `qnn.json`.

To run the steps manually with the currently configured accuracy level of `4`:

```powershell
# 1. Quantize
olive run --config gemma4_quantize.json

# 2. Export for either target
olive capture-onnx-graph `
    --model_name_or_path gemma4_quantized_hf `
    --use_mobius_builder `
    --precision fp32 `
    --int4_accuracy_level 4 `
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

`ov.json` starts from the `gemma4_onnx` export, fixes its patch
dimension and pooler depth, and produces fixed vision inputs
`pixel_values: [1, 2520, 768]`, `pixel_position_ids: [1, 2520, 2]`, and output
`image_features: [280, 1536]`. Valid image features precede padded rows.
Decoder, embedding, and audio export behavior is unchanged. The fixed-capacity
transformation is not applied during quantization or QNN optimization.

Olive writes the GenAI configuration, processor/tokenizer assets, and package
metadata automatically during capture.

`ov.json` applies `DynamicToFixedShape` to the vision input and output symbols,
replaces the vision pooler's dynamic `OneHot` depth with 280, and infers shapes
before converting quantized vision weights to QDQ. It then applies the existing
graph surgeries and runs `EPContextBinaryGenerator` for NPU. The
decoder pipeline also fixes KV-cache shapes and uses `StaticLLM` before
generating OpenVINO context binaries with shared weights.

The output is `gemma4_ov`, including the updated GenAI
configuration. Image inference requires a GenAI build supporting fixed-capacity
Gemma4 vision outputs with pipelined decoders, a matching OpenVINO EP plugin,
and the Intel NPU runtime. A non-pipelined decoder cannot use padded vision
outputs without additional runtime support.

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
as `--int4_accuracy_level` to the same `olive capture-onnx-graph` command
with `--use_mobius_builder`. No separate export JSON is used. If running the
steps manually, use the selected recipe's value in the capture command above.

The CLI option retains its existing `int4` name and is shared by Model Builder
and Mobius. With Mobius, it also applies to 8-bit `MatMulNBits`. The recipe
metadata key remains `matmul_nbits_accuracy_level`.

This overrides the activation-computation accuracy level
for all exported `MatMulNBits` operators, including the CPU embedding projection
and LM head retained after QNN conversion. It does not change weight bit widths
or `GatherBlockQuantized`, and is not a lossless numerical optimization.

The setting accepts integers `1` through `4`. Remove the setting or set it to
`null` to omit the capture argument and preserve Mobius's existing EP-specific
defaults. Explicit `0` is not accepted by the shared capture CLI.
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

`gemma4_quantize.json` sets the decoder's default to blockwise INT4 with
`group_size: 128` and
overrides `lm_head` with `{"bits": 8, "group_size": 128}`. This quantizes the
CPU LM head into supported blocks from the start; no post-export reblocking
is needed. Changing this override changes the LM head's quantization rather
than losslessly rearranging an existing quantized checkpoint.

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
