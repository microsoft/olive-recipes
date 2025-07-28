# Sample Guide

## Update sample.json

To make the sample work, you need to fill in the following properties

- MODEL_PATH: like Intel/bert-base-uncased-mrpc
- MODEL_TASK: like text-classification
- DS_NAME: like glue
- DS_SUBSET: like mrpc
- DS_SPLIT: like validation
- DATA_COLS: like [ "sentence1", "sentence2" ]
- FIXED_PARAMS: like [ "batch_size", "sequence_length" ]
- FIXED_VALUES: like [ 1, 128 ]

You could also adjust other parameters to suit your need:

- "execution_providers": [ "CPUExecutionProvider" ]: To other providers like QNNExecutionProvider. You need to run it on the matched device
- "max_length": 128 / "batch_size": 1: For static quantization, the input size should be fixed. Adjust these to match `FIXED_VALUES`
- "max_samples": 100: The number of samples used.

## Update model_project.comfig (optional)

Update `displayName` and `modelLink` to the one you used.

## Update sample.custom.config (optional)

This file is used to render the Run UX.
You could remove or add parameters match to your `sample.json`. Path update may be needed.

## Update inference_sample.ipynb (optional)

Write your own code to test the output model.
