# Phi-3.5 Model Optimization

This repository demonstrates the optimization of the [Microsoft Phi-3.5 Mini Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model using **post-training quantization (PTQ)** techniques.


### Quantization Python Environment Setup
Quantization is resource-intensive and requires GPU acceleration. In an x64 Python environment, install the required packages:

```bash
pip install -r requirements.txt

# AutoGPTQ: Install from source (stable package may be slow for weight packing)
# Disable CUDA extension build (not required)
# Linux
export BUILD_CUDA_EXT=0
# Windows
# set BUILD_CUDA_EXT=0

# Install AutoGPTQ from source
pip install --no-build-isolation git+https://github.com/PanQiWei/AutoGPTQ.git

# Install GptqModel from source
pip install --no-build-isolation git+https://github.com/ModelCloud/GPTQModel.git@5d2911a4b2a709afb0941d53c3882d0cd80b9649
```

### AOT Compilation Python Environment Setup
Model compilation using QNN Execution Provider requires a Python environment with onnxruntime-qnn installed. In a separate Python environment, install the required packages:

```bash
# Install Olive
pip install olive-ai==0.9.3

# Install ONNX Runtime QNN
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple "onnxruntime-qnn==1.22.2" --no-deps
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

```bash
olive run --config config.json
```

Olive will run the AOT compilation step in the **AOT Compilation Python Environment** specified in the config file using a subprocess. All other steps will run in the **Quantization Python Environment** natively.

✅ Optimized model saved in: `models/phi3_5-qnn/`

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.
