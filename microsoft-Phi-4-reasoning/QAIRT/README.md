# Phi-4 Reasoning Model Optimization

This directory demonstrates the optimization of the [Microsoft Phi-4 Reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) model using the QAIRT Pipeline API.

## Overview

This workflow uses the `QairtPipelinePass` to perform quantization, model transformation, DLC conversion, and HTP compilation in a single YAML-recipe-driven pass. The `QairtPipelinePass` recipe used within this directory is based on the [Qualcomm-distributed Jupyter notebook](https://qpm.qualcomm.com/#/main/tools/details/Tutorial_for_Phi4_Reasoning_14B_Compute) for Phi-4-reasoning which is available for download via QPM. Details on the QAIRT Pipeline API can be found in the [QAIRT Pipeline API documentation](https://docs.qualcomm.com/doc/80-87189-2/topic/guides.html?product=1601111740009302#pipeline-experimental).

After the pipeline pass, a `QairtEncapsulation` pass wraps the compiled DLC in an ONNX protobuf and exports a directory compatible with onnxruntime-genai.

## Metrics

| SoC | MMLU | TTFT (s) | Prefill TPS | Decode TPS |
|---|---|---|---|---|
| X2 Elite | 74.72 | 0.68 | 966.41 | 13.84 |
| X Elite | 74.45 | 1.80 | 367.30 | 4.65 |

## Previous Versions

Previous versions of this recipe used `QairtPreparationPass` and `QairtGenAIBuilderPass` in a script-based preparation workflow. This previous version of the recipe can be found [here](https://github.com/microsoft/olive-recipes/tree/4454b785c247826df77d5039ca07f1507a2f950a/microsoft-Phi-4-reasoning/QAIRT). Performance and accuracy metrics with this replacement recipe meet or exceed those measured with the original recipe.

## Requirements

**Validated host configuration:**
* Ubuntu 22.04
* Python 3.12.10
* Nvidia H100 80GB GPU, CUDA 12.8+
* QAIRT 2.48.40

With the preparation-time Python dependencies noted in requirements_host.txt  - other configurations may work but have not been verified.

Other configurations may work but have not been verified.

**Validated SoCs:**

The following SoC recipes were validated using target devices in the [Qualcomm Device Cloud](https://qdc.qualcomm.com/):

* Snapdragon X2 Elite
* Snapdragon X Elite

With the runtime Python dependencies noted in requirements_target.txt - other configurations may work but have not been verified.

## Preparation Instructions

1. Install host requirements

```bash
pip install -r requirements_host.txt
```

2. Use qairt-vm to install a non-default version of QAIRT and set QAIRT_SDK_ROOT

```bash
# Download QAIRT SDK 2.48.40
qairt-vm fetch -v 2.48.40 -d <download location>

# Set QAIRT_SDK_ROOT to the download location
# Note: No further QAIRT SDK installation steps are required when using qairt-dev
export QAIRT_SDK_ROOT=<download location>/qairt/<version>

# If using a non-system Python (e.g. uv, pyenv), expose the Python shared library
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
```

3. Run Olive recipe

For Snapdragon X2 Elite:
```bash
olive run --config x2_elite_config.json
```

For Snapdragon X Elite:
```bash
olive run --config x_elite_config.json
```

## Execution Instructions

The output of the above olive recipe is a directory validated on Snapdragon X2 Elite and Snapdragon X Elite with the following onnxruntime dependency versions (Python 3.12.10):

```bash
pip install -r requirements_target.txt
```

Please see the following script in the onnxruntime-genai repository for [an example of how to run this model directory](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).
