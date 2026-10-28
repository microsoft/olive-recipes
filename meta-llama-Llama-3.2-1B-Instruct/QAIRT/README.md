# Llama-3.2-1B-Instruct Model Optimization

This directory demonstrates the optimization of the [Meta Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model using the QAIRT Pipeline API.

## Overview

This workflow uses the `QairtPipelinePass` to perform quantization, model transformation, DLC conversion, and HTP compilation in a single YAML-recipe-driven pass. The `QairtPipelinePass`
recipe used within this directory is adapted from the QAIRT-tools validated Llama recipe for Snapdragon X Elite and X2 Elite laptop NPUs. Details on the QAIRT Pipeline API can be found in
the [QAIRT Pipeline API documentation](https://docs.qualcomm.com/doc/80-87189-2/topic/guides.html?product=1601111740009302#pipeline-experimental).

After the pipeline pass, a `QairtEncapsulation` pass wraps the compiled DLC in an ONNX protobuf and exports a directory compatible with onnxruntime-genai.

## Validated SoCs

The following SoC recipes were validated using target devices in the
[Qualcomm Device Cloud](https://qdc.qualcomm.com/):

* Snapdragon X Elite (SC8380XP)
* Snapdragon X2 Elite (SC8480XP)

Other configurations may work but have not been validated.

## Requirements

**Validated host configuration:**
* Ubuntu 22.04
* Python 3.10.12
* qairt-dev 0.12.0
* QAIRT 2.48.40

## Preparation Instructions

1. Install model-specific requirements

```bash
pip install -r requirements.txt
```

2. Use qairt-vm to install QAIRT 2.48.40 and set QAIRT_SDK_ROOT

```bash
# List available QAIRT SDK versions
qairt-vm fetch --list

# Download non-default version of QAIRT SDK
qairt-vm fetch -v 2.48.40 -d /path/to/qairt/sdk

# Set QAIRT_SDK_ROOT to the download location of the QAIRT SDK
# Note: No further QAIRT SDK installation steps are required when using qairt-dev
export QAIRT_SDK_ROOT=/path/to/qairt/sdk
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

The output of the above olive recipe is a directory validated with the following onnxruntime
dependency versions on Snapdragon X2 Elite and Snapdragon X Elite.

```bash
> python --version
3.12.10
> pip install onnxruntime==1.30.0 onnxruntime-genai==0.17.0 onnxruntime-qnn==2.7.0
```

Please see the following script in the onnxruntime-genai repository for
[an example of how to run this model directory](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).

## Performance

| SoC | PPL (Wiki) | TTFT (ms) | Prompt Processing Rate (tok/s) | Token Generation Rate (tok/s) |
|-----|------------|-----------|-------------------------------|-------------------------------|
| Snapdragon X Elite | TBD | TBD | TBD | TBD |
| Snapdragon X2 Elite | TBD | TBD | TBD | TBD |
