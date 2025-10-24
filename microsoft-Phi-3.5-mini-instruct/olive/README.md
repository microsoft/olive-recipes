# Phi-3.5-mini-instruct Mixed Precision Quantization

This recipe demonstrates how to use Olive to perform mixed precision (INT4/INT4) quantization, export to ONNX, and evaluate using lm-evaluation-harness. Please refer to the [Exploring Optimal Quantization Settings for Small Language Models with Olive](https://microsoft.github.io/Olive/blogs/quant-slms.html) for more details.

## Pre-requisites
Install Olive and other dependencies:

```bash
pip install -r requirements.txt
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
| Original (fp16) | 0.611         | 0.834    | 0.688 | 0.769     | 0.608     | 0.460      | 7.142         |
| Mixed           | 0.617         | 0.832    | 0.680 | 0.765     | 0.602     | 0.482      | 2.453         |
