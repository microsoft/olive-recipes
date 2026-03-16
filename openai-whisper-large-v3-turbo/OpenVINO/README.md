# Whisper Large V3 Turbo Optimization for Intel® NPU

This example demonstrates how to export and optimize OpenAI's Whisper Large V3 Turbo model to OpenVINO IR encapsulated ONNX model format using Olive with Intel® optimum-cli passes for Intel® NPU

## Model Information

- **Model**: `openai/whisper-large-v3-turbo`
- **Architecture**: 4 decoder layers (compared to 32 in whisper-large-v3)
- **Target Hardware**: Intel® NPU
- **Output Format**: OpenVINO IR encapsulated ONNX model

## Prerequisites

Install the required packages:

```bash
python -m pip install "git+https://github.com/microsoft/olive.git@main#egg=olive-ai[openvino]"
```

## Quick Start

### Basic Conversion

```bash
python convert_whisper_to_ovir.py
```

This will convert `openai/whisper-large-v3-turbo` with default settings, which are FP16 weights, NPU weight sharing disabled and without reshaping models at conversion time.

### Command Line Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `-m, --model` | HuggingFace model ID | `openai/whisper-large-v3-turbo` |
| `-w, --weight-format` | Weight compression format: `fp16`, `int8`, `int4` | `fp16` |
| `--enable_npu_ws` | Enable NPUW weight sharing between prefill & generate models | `False` |
| `--reshape` | Reshape models at conversion time instead of using provider options | `False` |

### Examples

**Convert with INT4 quantization:**

```bash
python convert_whisper_to_ovir.py -w int4
```

**Convert with INT8 quantization and NPU weight sharing:**

```bash
python convert_whisper_to_ovir.py -w int8 --enable_npu_ws True
```

**Convert Whisper Base instead:**

```bash
python convert_whisper_to_ovir.py -m openai/whisper-base
```

## Output Structure

After conversion, the model will be saved to `model/whisper-large-v3-turbo-fp16-ov/` (if using default FP16 option) with:

```text
model/whisper-large-v3-turbo-fp16-ov/
|-- audio_processor_config.json
|-- genai_config.json
|-- openvino_decoder_model.bin
|-- openvino_decoder_model.onnx
|-- openvino_decoder_model.xml
|-- openvino_detokenizer.bin
|-- openvino_detokenizer.xml
|-- openvino_encoder_model.bin
|-- openvino_encoder_model.onnx
|-- openvino_encoder_model.xml
|-- openvino_tokenizer.bin
|-- openvino_tokenizer.xml
|-- ...
```

## Default template config Files

- `whisper_large_v3_turbo_default_ov_npu.json` - Olive Intel® optimum conversion config file
- `whisper_large_v3_turbo_encapsulate.json` - Olive OVIR ONNX encapsulation config file
- `audio_processor_config_default.json` - Audio processor config file
