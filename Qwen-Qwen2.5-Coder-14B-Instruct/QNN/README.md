# Qwen2.5-Coder-14B-Instruct Model Optimization

This repository demonstrates the optimization of the [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) model using **post-training quantization (PTQ)** techniques.

### Quantization Python Environment Setup
Quantization is resource-intensive and requires GPU acceleration. In an x64 Python environment, install the required packages:

```bash
pip install -r requirements.txt

# Disable CUDA extension build (not required)
# Linux
export BUILD_CUDA_EXT=0
# Windows
# set BUILD_CUDA_EXT=0

# Install GptqModel from source
pip install --no-build-isolation git+https://github.com/CodeLinaro/GPTQModel.git@rel_4.2.5
```

### AOT Compilation Python Environment Setup
Model compilation using QNN Execution Provider requires a Python environment with onnxruntime-qnn installed. In a separate Python environment, install the required packages:

```bash
# Install Olive
pip install olive-ai==0.13.0

# Install ONNX Runtime QNN
pip install onnxruntime==1.26.0
pip install onnxruntime-qnn==2.4.0
```

Replace `/path/to/qnn/env/bin` in the config file with the path to the directory containing your QNN environment's Python executable. This path can be found by running the following command in the environment:

```bash
# Linux
command -v python
# Windows
# where python
```

This command will return the path to the Python executable. Set the parent directory of the executable as the `/path/to/qnn/env/bin` in the config file.

### Run the Quantization + Compilation Config
Activate the **Quantization Python Environment** and run the workflow.

For Snapdragon X Elite:

```bash
olive run --config x_elite_config.json
```

For Snapdragon X2 Elite:

```bash
olive run --config x2_elite_config.json
```

Olive will run the AOT compilation step in the **AOT Compilation Python Environment** specified in the config file using a subprocess. All other steps will run in the **Quantization Python Environment** natively.

Optimized model saved in: `models/qwen2.5_coder_14B_instruct/`

> If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.

> If the Static Quantization (SQ) pass fails with `Failed to allocate memory buffer of size...`, rerun the command without clearing the cache. Olive will resume from the last completed step and the pass will succeed.


### Calibration Dataset Experiment Using a Coding Dataset

The impact of the calibration dataset was evaluated using the
[LiveCodeBench benchmark](https://github.com/LiveCodeBench) on the
Qwen2.5-Coder-14B-Instruct model.

Two calibration datasets were evaluated:

| Calibration dataset | LiveCodeBench score |
|---|---:|
| WikiText-2 | 26.5 |
| [Evol-Instruct-Code-80k-v1-rogery-2k-sampled-20250426](https://huggingface.co/datasets/chnug/Evol-Instruct-Code-80k-v1-rogery-2k-sampled-20250426) | **30.434** |

The coding-focused calibration dataset produced a higher observed
LiveCodeBench score than WikiText-2. This indicates that calibration data
aligned with the model's target coding workload may be more effective than
general-domain text for this model.

> **Note:** The scores above are the results observed in this evaluation.
> Results may vary depending on the preprocessing steps, quantization
> settings, evaluation configuration, model runtime, and LiveCodeBench
> version.

For the coding-based calibration experiment, download the
[Evol-Instruct-Code-80k-v1-rogery-2k-sampled-20250426](https://huggingface.co/datasets/chnug/Evol-Instruct-Code-80k-v1-rogery-2k-sampled-20250426)
dataset from Hugging Face.

Convert the dataset
into the format required for quantization by combining the `instruction` and
`output` fields into a single `text` field using the following preprocessing
function:

```python
def convert_to_text(example):
    instruction = str(example["instruction"]).strip()
    output = str(example["output"]).strip()

    return {
        "text": (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
            f"{output}"
        )
    }

#### This recipe was last validated with the versions specified above.