# Qwen3-0.6B Model Optimization

This repository demonstrates the optimization of the [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model using **post-training quantization (PTQ)** techniques. This code was originally run on Ubuntu 22.04, use the same OS for maximum compatibility. 


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
pip install onnxruntime-qnn==2.3.0 --no-deps
```

Replace `/path/to/qnn/env/bin` in [config.json](config.json) with the path to the directory containing your QNN environment's Python executable. This path can be found by running the following command in the environment:

```bash
# Linux
command -v python
# Windows
# where python
```

This command will return the path to the Python executable. Set the parent directory of the executable as the `/path/to/qnn/env/bin` in the config file.

### Run the Quantization + Compilation Config
Activate the **Quantization Python Environment** and run the workflow:

### Change the soc_model param in config.json file corresponding to the target platform

```bash
olive run --config config.json
```

Olive will run the AOT compilation step in the **AOT Compilation Python Environment** specified in the config file using a subprocess. All other steps will run in the **Quantization Python Environment** natively.

Optimized model saved in: `models/Qwen3-0.6B/`

> If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.