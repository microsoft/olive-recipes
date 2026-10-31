# Phi-4-mini-instruct Model Optimization

This directory demonstrates the optimization of the [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) model using various AIMET quantization techniques.

## Overview

After quantization, the QAIRT GenAIBuilder API is utilized to apply additional model transformations, perform conversion, and compile the model for execution on the HTP backend.

Finally, a prepared QAIRT DLC is encapsulated in an ONNX protobuf and exported to a directory compatible with onnxruntime-genai.

## Requirements

**Validated host configuration:**
* Ubuntu 22.04
* Python 3.10.12
* qairt-dev 0.11.0
* QAIRT 2.48.40

**Validated target configuration:**
* HTP backend on SC8380XP, SC8480XP

## Preparation Instructions

1. Authenticate with Hugging Face

```bash
huggingface-cli login  # Recommended: stores credentials securely, avoids shell history
# Alternative: export HF_TOKEN=<your_hugging_face_token>
```

2. Prepare Environment

```bash
pip install --no-deps -r requirements.txt
```

3. Use qairt-vm to install a non-default version of QAIRT and set QAIRT_SDK_ROOT

```bash
# List available QAIRT SDK versions
qairt-vm fetch --list

# Download non-default version of QAIRT SDK
qairt-vm fetch -v <version>

# Set QAIRT_SDK_ROOT to download location of QAIRT SDK
# By default, /opt/qcom/aistack/qairt/<version>
# Note: No further QAIRT SDK installation steps are required when using qairt-dev
export QAIRT_SDK_ROOT=/path/to/qairt/sdk
```

4. Run Olive recipe

```bash
# For X Elite:
olive run --config htp_sc8380xp.json
```

```bash
# For X2 Elite:
olive run --config htp_sc8480xp.json
```

## Execution Instructions

The output of the above olive recipe is a directory compatible with the following versions of onnxruntime-genai and onnxruntime-qnn.

```bash
pip install onnxruntime-genai>=0.13
pip install onnxruntime-qnn>=2.4.0
```

Please see the following script in the onnxruntime-genai repository for [an example of how to run this model directory](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).

## Known Issues

### `AttributeError: module 'pydantic._internal._typing_extra' has no attribute 'add_module_globals'`

This error can occasionally occur on the first invocation of the recipe. If encountered, re-running the recipe is sufficient as a workaround.
