# Phi-4 Reasoning Model Optimization

This directory demonstrates the optimization of the [Microsoft Phi-4 Reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) model using various AIMET quantization techniques.

## Overview

This workflow utilizes a Phi-4 Reasoning script to perform quantization based on the [Qualcomm-distributed Jupyter notebook](https://qpm.qualcomm.com/#/main/tools/details/Tutorial_for_Phi4_Reasoning_14B_Compute) for Phi-4-reasoning which is available for download via QPM.

After quantization, the QAIRT GenAIBuilder API is utilized to apply additional model transformations, perform conversion, and compile the model for execution on the HTP backend.

Finally, a prepared QAIRT DLC is encapsulated in an ONNX protobuf and exported to a directory compatible with onnxruntime-genai.

## Requirements

This workflow has been tested using the following host configuration:
* Python 3.10
* qairt-dev 0.5.0
* QAIRT 2.45.40

Further, this workflow has been tested on the following target configurations:
* HTP backend on SC8480XP

## Preparation Instructions

1. Install olive[qairt]

```bash
pip install olive[qairt]
pip list | grep qairt-dev  # Ensure the proper qairt-dev version  was installed
pip install qairt-dev[onnx]==<version>  # Install the proper qairt-dev version, if not installed
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

```bash
olive run --config htp_sc8480xp.json
```

## Execution Instructions

The output of the above olive recipe is a directory compatible with the following versions of onnxruntime-genai and onnxruntime-qnn.

```bash
pip install onnxruntime-genai>=0.13
pip install onnxruntime-qnn>=2.1.0
```

Please see the following script in the onnxruntime-genai repository for [an example of how to run this model directory](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).
