# Phi-4 Reasoning Model Optimization

This directory demonstrates the optimization of the [Microsoft Phi-4 Reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) model using various AIMET quantization techniques.

## Overview

This workflow utilizes a Phi-4 Reasoning script to perform quantization based on the [Qualcomm-distributed Jupyter notebook](https://qpm.qualcomm.com/#/main/tools/details/Tutorial_for_Phi4_Reasoning_14B_Compute) for Phi-4-reasoning which is available for download via QPM.

After quantization, the QAIRT GenAIBuilder API is utilized to apply additional model transformations, perform conversion, and compile the model for execution on the HTP backend.

Finally, a prepared QAIRT DLC is encapsulated in an ONNX protobuf and exported to a directory compatible with onnxruntime-genai.

## Requirements

This workflow has been tested using the following host configuration:
    * Python 3.10
    * QAIRT 2.45.0

Further, this workflow has been tested on the following target configurations:
    * HTP backend on SC8380XP
    * HTP backend on SC8480XP

## Preparation Instructions

## Execution Instructions

