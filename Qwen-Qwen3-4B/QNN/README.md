# Qwen3-4B Model Optimization

This repository demonstrates the optimization of the [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) model using **post-training quantization (PTQ)** techniques. 

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
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
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

Optimized model saved in: `models/qwen3_4B/`

> If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.

> If the Static Quantization (SQ) pass fails with `Failed to allocate memory buffer of size...`, rerun the command without clearing the cache. Olive will resume from the last completed step and the pass will succeed.