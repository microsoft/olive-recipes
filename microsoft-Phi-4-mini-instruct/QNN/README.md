# Phi-4 Mini Instruct Model Optimization

This repository demonstrates the optimization of the [Microsoft Phi-4 Mini Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into two main workflows:

## Run Olive workflow with CLI

### Quantization Python Environment Setup

Quantization is resource-intensive and requires GPU acceleration. In an x64 Python environment with Olive installed, install the required packages:

```bash
# Install common dependencies
pip install -r requirements.txt

# Install ONNX Runtime GPU packages
pip install "onnxruntime-genai-cuda>=0.9.0"

# AutoGPTQ: Install from source (stable package may be slow for weight packing)
# Disable CUDA extension build (not required)
# Linux
export BUILD_CUDA_EXT=0
# Windows
# set BUILD_CUDA_EXT=0

# Install AutoGPTQ from source
pip install --no-build-isolation git+https://github.com/PanQiWei/AutoGPTQ.git
```

### AOT Compilation Python Environment Setup

Model compilation using QNN Execution Provider requires a Python environment with onnxruntime-qnn installed. In a separate Python environment with Olive installed, install the required packages:

```bash
# Install ONNX Runtime QNN
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
```

This qnn env path can be found by running the following command in the environment:

```bash
# Linux
command -v python
# Windows
# where python
```

This command will return the path to the Python executable.

### Run the Quantization + Compilation CLI

Activate the **Quantization Python Environment**, replace the `/path/to/qnn/env/bin` by the actual path from previous step, and run the CLI:

```bash
olive optimize -m microsoft/Phi-4-mini-instruct --provider QNNExecutionProvider --device npu --precision int4 --num_split 4 --enable_aot --qnn_env_path </path/to/qnn/env/bin> --surgeries RemoveRopeMultiCache,AttentionMaskToSequenceLengths,SimplifiedLayerNormToL2Norm --act_precision uint16 --use_qdq_format --log_level 1
```

Olive will run the AOT compilation step in the **AOT Compilation Python Environment** using a subprocess. All other steps will run in the **Quantization Python Environment** natively.

✅ Optimized model saved in: `optimized-model/`

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.

## Run Olive workflow with docker

Optimizing the model with Docker will simplify the installation process, so the Dockerfile we built already sets up the environments for you to use out of the box.

### Install dependencies

install the Olive with required packages:

```bash
pip install olive-ai[docker]
```

### Run the Olive workflow

```bash
python -m olive run --config phi4_mini_qnn_docker.json
```

The output models will be saved in `models/phi4-mini-qnn-docker`. Simply update `output_dir` field in the config file to customize your own output folder.

## Run Olive workflow with QNN-GPU

### QNN-GPU: Run the Quantization Config

Running QNN-GPU configs requires features and fixes that are not available in the released Olive version. To ensure compatibility, please install Olive directly from the source at the required commit:

```bash
pip install git+https://github.com/microsoft/Olive.git@da24463e14ed976503dc5871572b285bc5ddc4b2
```

If you previously installed Olive via PyPI, please uninstall it first and then use the above commit to install:

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

✅ Optimized model saved in: `models/phi4-mini-instruct/`

### QNN-GPU: Run the Context Binary Compilation Config

Replace `/path/to/model/` in [config_gpu_ctxbin.json](config_gpu_ctxbin.json) with the output path generated from above step.

Activate the **AOT Python Environment** and run the workflow:

```bash
olive run --config config_gpu_ctxbin.json
```

✅ Optimized model saved in: `models/phi4-mini-instruct/`

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.
