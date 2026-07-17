# Phi-4 Reasoning Model Optimization

This directory demonstrates the optimization of the [Microsoft Phi-4 Reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) model using the QAIRT Pipeline API.

## Overview

This workflow uses the `QairtPipelinePass` to perform quantization, model transformation, DLC conversion, and HTP compilation in a single YAML-recipe-driven pass.

After the pipeline pass, a `QairtEncapsulation` pass wraps the compiled DLC in an ONNX protobuf and exports a directory compatible with onnxruntime-genai.

## Requirements

**Validated host configuration:**
* Ubuntu 22.04
* Python 3.10.12
* qairt-dev 0.9.0
* QAIRT 2.45.40

**Validated target configurations:**
* HTP backend on SC8480XP
* HTP backend on SC8380XP

Other configurations may work but have not been validated.

## Preparation Instructions

1. Install olive-ai[qairt]

```bash
pip install olive-ai[qairt]
pip list | grep qairt-dev  # Ensure the proper qairt-dev version was installed
pip install qairt-dev[onnx]==0.9.0  # Install the proper qairt-dev version, if not installed
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

3. Install model-specific requirements

```bash
pip install -r requirements.txt
```

4. Run Olive recipe

For SC8480XP:
```bash
olive run --config htp_sc8480xp.json
```

For SC8380XP:
```bash
olive run --config htp_sc8380xp.json
```

## Execution Instructions

The output of the above olive recipe is a directory compatible with the following versions of onnxruntime-genai and onnxruntime-qnn.

```bash
pip install onnxruntime-genai>=0.13
pip install onnxruntime-qnn>=2.1.0
```

Please see the following script in the onnxruntime-genai repository for [an example of how to run this model directory](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).
