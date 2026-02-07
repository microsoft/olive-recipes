# Model Optimization and Quantization for AMD NPU

This folder contains sample Olive configuration to optimize Phi-4 models for AMD NPU.

## ✅ Supported Models and Configs

| Model Name (Hugging Face)                          | Config File Name                  |
|:---------------------------------------------------|:----------------------------------|
| `microsoft/Phi-4-mini-instruct`                    | `Phi-4-mini-instruct_quark_vitisai_llm.json`  |

## **Run the Quantization Config**

### **Quark quantization**

For LLMs - follow the below commands to generate the optimized model for VitisAI Execution Provider.

**Platform Support:**
- ✅ **Linux with ROCm** - Supported
- ✅ **Linux with CUDA** - Supported
- ✅ **Windows with CUDA** - Supported
- ✅ **Windows with CPU** - Supported (quantization will be slower)
- ⏳ **Windows with ROCm** - Planned for future release

For more details about quark, see the [Quark Documentation](https://quark.docs.amd.com/latest/)

#### **Create a Python 3.10 conda environment and run the below commands**
```bash
conda create -n olive python=3.12
conda activate olive
```

```bash
cd Olive
pip install -e .
pip install -r requirements.txt
```

#### **Install VitisAI LLM dependencies**

```bash
cd olive-recipes/microsoft-Phi-4-mini-instruct/VitisAI
pip install --force-reinstall -r requirements_vitisai_llm.txt

# Note: If you're running model generation on a Windows system, please uncomment the following line in requirements_vitisai_llm.txt:
# --extra-index-url=https://pypi.amd.com/simple
# model-generate==1.5.1
```

Make sure to install the correct version of PyTorch before running quantization. If using AMD GPUs, update PyTorch to use ROCm-compatible PyTorch build. For example see the below commands

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

python -c "import torch; print(torch.cuda.is_available())" # Must return `True`
```
#### **Generate optimized LLM model for VitisAI NPU**
Follow the above setup instructions, then run the below command to generate the optimized LLM model for VitisAI EP

```bash
# Phi-4-mini-instruct
olive run --config Phi-4-mini-instruct_quark_vitisai_llm.json
```

✅ Optimized model saved in: `models/Phi-4-mini-instruct-vai/`
> **Note:** Output model is saved in `output_dir` mentioned in the json files.
