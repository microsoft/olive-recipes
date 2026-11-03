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
* qairt-dev 0.14.0
* QAIRT 2.48.40
* NVIDIA GPU with ≥80 GB VRAM (e.g. H100 80GB) — required for AdaScale and quantization stages

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

2. Install olive-ai[qairt] and qairt-dev

```bash
pip install olive-ai[qairt]
pip install qairt-dev[pipeline,onnx]==0.14.0
```

3. Install model-specific requirements

PyTorch must be installed with CUDA support. Use the PyTorch index URL matching your CUDA version (the recipe was validated with CUDA 12.8):

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

4. Run Olive recipe

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required to prevent CUDA allocator fragmentation during the quantization export phase.

For Snapdragon X2 Elite:
```bash
QAIRT_SDK_ROOT=<path/to/qairt/sdk> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True olive run --config x2_elite_config.json
```

For Snapdragon X Elite:
```bash
QAIRT_SDK_ROOT=<path/to/qairt/sdk> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True olive run --config x_elite_config.json
```

## Execution Instructions

The output of the above olive recipe is a directory validated with the following onnxruntime dependency versions on Snapdragon X2 Elite and Snapdragon X Elite.

```bash
> python --version
3.12.10
> pip install onnxruntime==1.24.2 onnxruntime-genai==0.18.0 onnxruntime-qnn==2.4.0
```

Please see the following script in the onnxruntime-genai repository for [an example of how to run this model directory](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).
