# Llama-3.2-1B-Instruct Mixed Precision Quantization

This recipe demonstrates how to use Olive to perform mixed precision (INT4/INT4) quantization, export to ONNX, and evaluate using lm-evaluation-harness. Please refer to the [Exploring Optimal Quantization Settings for Small Language Models with Olive](https://microsoft.github.io/Olive/blogs/quant-slms.html) for more details.

## Pre-requisites
Install Olive and other dependencies:

```bash
pip install -r requirements-mixed.txt
```

## Run
To run the mixed precision quantization recipe, execute the following command:

```bash
olive run --config mixed.json
```

**Note**: Evaluation requires a machine with CUDA enabled GPU. If you don't have a GPU, you can skip the evaluation step by modifying the `mixed.json` file to remove the `"evaluator": "evaluator"` line.

## Results
| model           | arc_challenge | arc_easy | mmlu  | hellaswag | mmlu_stem | openbookqa | model_size_gb |
|-----------------|---------------|----------|-------|-----------|-----------|------------|---------------|
| Original (fp16) | 0.381         | 0.632    | 0.460 | 0.607     | 0.389     | 0.346      | 2.807         |
| Mixed           | 0.376         | 0.625    | 0.458 | 0.598     | 0.406     | 0.344      | 1.361         |
