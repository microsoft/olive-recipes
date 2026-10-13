# Llama-3.1-8B-Instruct Model Optimization

This directory demonstrates the optimization of the [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model using the QAIRT Pipeline API.

## Overview

This workflow uses the `QairtPipelinePass` to perform quantization, model transformation, DLC conversion, and HTP compilation in a single YAML-recipe-driven pass. Details on the QAIRT Pipeline API can be found in the [QAIRT Pipeline API documentation](https://docs.qualcomm.com/doc/80-87189-2/topic/guides.html?product=1601111740009302#pipeline-experimental).

After the pipeline pass, a `QairtEncapsulation` pass wraps the compiled DLC in an ONNX protobuf and exports a directory compatible with onnxruntime-genai.

## Previous Versions

Previous versions of this recipe used `QairtPreparation` and `QairtGenAIBuilder` in a script-based preparation workflow. This previous version of the recipe can be found [here](https://github.com/microsoft/olive-recipes/tree/4e027c56a69d610208aef2818e5f54ce93a83cbd/meta-llama-Llama-3.1-8B-Instruct/QAIRT).

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

1. Authenticate with Hugging Face — this recipe uses the gated `meta-llama/Llama-3.1-8B-Instruct` model. Request access on Hugging Face, then authenticate:

```bash
huggingface-cli login  # Recommended: stores credentials securely, avoids shell history
# Alternative: export HF_TOKEN=<your_hugging_face_token>
```

2. Install olive-ai[qairt]

```bash
pip install olive-ai[qairt]
pip list | grep qairt-dev  # Ensure the proper qairt-dev version was installed
pip install qairt-dev[pipeline,onnx]==0.11.0  # Install the proper qairt-dev version, if not installed
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
