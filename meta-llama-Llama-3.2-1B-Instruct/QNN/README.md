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
pip install olive-ai==0.11.0

# Install ONNX Runtime QNN
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple "onnxruntime-qnn==1.22.2" --no-deps
```

### QNN-GPU: Run the Quantization Config

QNN-GPU configs require Olive >= 0.11.0 (already satisfied by the pinned version in [requirements.txt](requirements.txt)).

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

The `StaticLLM` pass in [config_gpu.json](config_gpu.json) defaults `context_iterator_models` to `true`, which generates **both** the AR1 (`iterator`, sequence length 1) and AR128 (`context`, sequence length = `context_length`) static models in this single run, and updates `genai_config.json` with a `decoder.pipeline` entry covering both components.

To generate only a single static model instead (sequence length = `context_length`), set `"context_iterator_models": false`.

### QNN-GPU: Run the Context Binary Compilation Config

These configs are shared across the Qwen2.5-1.5B-Instruct, Llama-3.1-8B-Instruct, and Phi-3.5-mini-instruct QNN-GPU recipes as well. Pick the one that matches how `context_iterator_models` was set for [config_gpu.json](config_gpu.json) above:

- **Single model** (`context_iterator_models: false`, one static model): use [config_gpu_ctxbin.json](config_gpu_ctxbin.json). Replace `/path/to/model/`, `additional_files` list, and `output_dir` in that config with the corresponding paths generated from the above step.
- **Composite model** (`context_iterator_models: true`, the default — AR1 `iterator` + AR128 `context`): use [config_gpu_composite_ctxbin.json](config_gpu_composite_ctxbin.json), which loads both static ONNX models as a `CompositeModel` and sets `weight_sharing: true` on the `EPContextBinaryGenerator` pass so weights shared between the two models aren't duplicated in the compiled context binaries. Replace the `context.onnx`/`iterator.onnx` paths under `model_components`, the `additional_files` list, and `output_dir` with the paths generated from the above step.

Activate the **AOT Python Environment** and run the workflow:

```bash
olive run --config config_gpu_ctxbin.json
# or, for the composite (context + iterator) model
olive run --config config_gpu_composite_ctxbin.json
```

✅ Optimized model saved in: `models/llama3.2_1b_Instruct/`

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.
