# Qwen3-8B-Instruct Model Optimization

This directory demonstrates the optimization of the [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) model using various AIMET quantization techniques.

## Overview

This workflow utilizes a Qwen3-8B script to perform quantization based on the [Qualcomm-distributed Jupyter notebook](https://qpm.qualcomm.com/#/main/tools/details/Tutorial_for_Qwen3) for Qwen3-8B (v1.0.1.260219) which is available for download via QPM.

After quantization, the QAIRT GenAIBuilder API is utilized to apply additional model transformations, perform conversion, and compile the model for execution on the HTP backend.

Finally, a prepared QAIRT DLC is encapsulated in an ONNX protobuf and exported to a directory compatible with onnxruntime-genai.

## Requirements

**Validated host configuration:**
* Ubuntu 22.04
* Python 3.12.0
* qairt-dev 0.11.0
* QAIRT 2.48.40 / 2.50.40

**Validated target configuration:**
* HTP backend on SC8480XP
* HTP backend on SC8380XP

Other configurations may work but have not been validated.

## Preparation Instructions

1. Install olive-ai[qairt]

```bash
pip install --no-build-isolation git+https://github.com/microsoft/olive.git@d4b9c1b63e4874cacc9d6efb2aa649de7f2d2710
pip list | grep qairt-dev  # Ensure the proper qairt-dev version  was installed
pip install qairt-dev[onnx]==0.11.0  # Install the proper qairt-dev version, if not installed
```

2. (Optional) Use qairt-vm to install a non-default version of QAIRT and set QAIRT_SDK_ROOT

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

4. Install model-specific requirements

```bash
pip install -r requirements.txt --no-deps
```

5. Run Olive recipe

```bash
# For X Elite:
olive run --config htp_sc8380xp.json

# For X2 Elite:
olive run --config htp_sc8480xp.json
```

## Execution Instructions

The output of the above olive recipe is a directory compatible with the following versions of onnxruntime-genai and onnxruntime-qnn.

For QAIRT==2.48.40
```bash
pip install onnxruntime-genai==0.15.2
pip install onnxruntime-qnn==2.4.0
```

For QAIRT==2.50.40
```bash
pip install onnxruntime-genai==0.15.2
pip install onnxruntime-qnn==2.6.0
```

Please see the following script in the onnxruntime-genai repository for [an example of how to run this model directory](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).

## Known Issues

### `AttributeError: module 'pydantic._internal._typing_extra' has no attribute 'add_module_globals'`

This error can occasionally occur on the first invocation of the recipe. If encountered, re-running the recipe is sufficient as a workaround.
