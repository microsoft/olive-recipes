# Llama-3.2-1B-Instruct Model Optimization

This repository demonstrates the optimization of the [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these workflows:


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

### QNN-GPU: Run the Quantization Config

Running QNN-GPU configs requires features and fixes that are not available in the released Olive version 0.9.3.
To ensure compatibility, please install Olive directly from the source at the required commit:

```bash
pip install git+https://github.com/microsoft/Olive.git@da24463e14ed976503dc5871572b285bc5ddc4b2
```

If you previously installed Olive via PyPI or pinned it to version 0.9.3, please uninstall it first and then use the above
commit to install:

```bash
pip uninstall olive-ai
```

Replace `/path/to/qnn/env/bin` in [config_gpu.json](config_gpu.json) with the path to the directory containing your QNN environment's Python executable. This path can be found by running the following command in the environment:

```bash
# Linux
command -v python
# Windows
# where python
```

This command will return the path to the Python executable. Set the parent directory of the executable as the `/path/to/qnn/env/bin` in the config file.

Activate the **Quantization Python Environment** and run the workflow:

```bash
olive run --config config_gpu.json
```

✅ Optimized model saved in: `models/llama3.2_1b_Instruct/`

### QNN-GPU: Run the Context Binary Compilation Config

Replace `/path/to/model/` in [config_gpu_ctxbin.json](config_gpu_ctxbin.json) with the output path generated from above step.

Activate the **AOT Python Environment** and run the workflow:

```bash
olive run --config config_gpu_ctxbin.json
```

✅ Optimized model saved in: `models/llama3.2_1b_Instruct/`

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.
