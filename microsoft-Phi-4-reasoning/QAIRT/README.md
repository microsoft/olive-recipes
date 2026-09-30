# Phi-4 Reasoning Model Optimization

This directory demonstrates the optimization of the [Microsoft Phi-4 Reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) model using the QAIRT Pipeline API.

## Overview

This workflow uses the `QairtPipelinePass` to perform quantization, model transformation, DLC conversion, and HTP compilation in a single YAML-recipe-driven pass. The `QairtPipelinePass` recipe used within this directory is based on the [Qualcomm-distributed Jupyter notebook](https://qpm.qualcomm.com/#/main/tools/details/Tutorial_for_Phi4_Reasoning_14B_Compute) for Phi-4-reasoning which is available for download via QPM. Details on the QAIRT Pipeline API can be found in the [QAIRT Pipeline API documentaton](https://docs.qualcomm.com/doc/80-87189-2/topic/guides.html?product=1601111740009302#pipeline-experimental).

After the pipeline pass, a `QairtEncapsulation` pass wraps the compiled DLC in an ONNX protobuf and exports a directory compatible with onnxruntime-genai.

## Previous Versions

Previous versions of this recipe used `QairtPreparationPass` and `QairtGenAIBuilderPass` in a script-based preparation workflow. This previous version of the recipe can be found [here](https://github.com/microsoft/olive-recipes/tree/4454b785c247826df77d5039ca07f1507a2f950a/microsoft-Phi-4-reasoning/QAIRT). Performance and accuracy metrics with this replacement recipe meet or exceed those measured with the original recipe.

## Requirements

**Validated host configuration:**
* Ubuntu 22.04
* Python 3.10.12
* qairt-dev 0.11.0
* QAIRT 2.45.40

**Validated SoCs:**

The following SoC recipes were validated using target devices in the [Qualcomm Device Cloud](https://qdc.qualcomm.com/):

* Snapdragon X2 Elite
* Snapdragon X Elite

Other configurations may work but have not been validated.

## Preparation Instructions

1. Install olive-ai[qairt]

```bash
pip install olive-ai[qairt]
pip list | grep qairt-dev  # Ensure the proper qairt-dev version was installed
pip install qairt-dev[pipeline,onnx]==0.11.0  # Install the proper qairt-dev version, if not installed
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

For Snapdragon X2 Elite:
```bash
olive run --config x2_elite_config.json
```

For Snapdragon X Elite:
```bash
olive run --config x_elite_config.json
```

## Execution Instructions

The output of the above olive recipe is a directory validated with the following onnxruntime dependency versions on Snapdragon X2 Elite and Snapdragon X Elite.

```bash
> python --version
3.12.10
> pip install onnxruntime==1.24.2 onnxruntime-genai==0.13.0 onnxruntime-qnn==2.1.0
```

Please see the following script in the onnxruntime-genai repository for [an example of how to run this model directory](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).
