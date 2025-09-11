# Qwen2.5-1.5B-Instruct Model Optimization

This repository demonstrates the optimization of the [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model using **post-training quantization (PTQ)** techniques. 


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
olive optimize -m Qwen/Qwen2.5-1.5B-Instruct --provider QNNExecutionProvider --device npu --precision int4 --num_split 4 --enable_aot --qnn_env_path </path/to/qnn/env/bin> --surgeries RemoveRopeMultiCache,AttentionMaskToSequenceLengths,SimplifiedLayerNormToL2Norm --act_precision uint16 --use_qdq_format
```

Olive will run the AOT compilation step in the **AOT Compilation Python Environment** using a subprocess. All other steps will run in the **Quantization Python Environment** natively.

✅ Optimized model saved in: `optimized-model/`

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.
