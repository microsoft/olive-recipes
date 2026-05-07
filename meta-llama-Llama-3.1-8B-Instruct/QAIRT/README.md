# Llama3.1-8B-Instruct Model Optimization

This directory demonstrates the optimization of the [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) model using various AIMET quantization techniques.

## Overview

This workflow utilizes a Llama3.1-8B-Instruct script to perform quantization based on the [Qualcomm-distributed Jupyter notebook](https://qpm.qualcomm.com/#/main/tools/details/Tutorial_for_Llama3_1_Compute) for Llama3.1-8B-Instruct (v1.0.1.260219) which is available for download via QPM.

After quantization, the QAIRT GenAIBuilder API is utilized to apply additional model transformations, perform conversion, and compile the model for execution on the HTP backend.

Finally, a prepared QAIRT DLC is encapsulated in an ONNX protobuf and exported to a directory compatible with onnxruntime-genai.

## Requirements

* Python 3.10.12
* Ubuntu 22.04
* qairt-dev 0.5.0
* QAIRT 2.45.40

Other configurations may work but have not been validated.

Further, this workflow has been tested on the following target configurations:
* HTP backend on SC8480XP

## Preparation Instructions

1. Authenticate with Hugging Face — this recipe uses the gated `meta-llama/Llama-3.1-8B-Instruct` model. Request access on Hugging Face, then authenticate:

```bash
huggingface-cli login  # Recommended: stores credentials securely, avoids shell history
# Alternative: export HF_TOKEN=<your_hugging_face_token>
```

2. Install olive-ai[qairt]

```bash
pip install olive-ai[qairt]
pip list | grep qairt-dev  # Ensure the proper qairt-dev version  was installed
pip install qairt-dev[onnx]==<version>  # Install the proper qairt-dev version, if not installed
```

3. (Optional) Use qairt-vm to install a non-default version of QAIRT and set QAIRT_SDK_ROOT

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
pip install -r requirements.txt
pip install torch==2.1.0
```

5. Run Olive recipe

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

## Known Issues

### `AttributeError: module 'pydantic._internal._typing_extra' has no attribute 'add_module_globals'`

This error can occasionally occur on the first invocation of the recipe. If encountered, re-running the recipe is sufficient as a workaround.
