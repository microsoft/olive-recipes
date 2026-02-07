# Model Optimization and Quantization for AMD NPU

This folder contains sample Olive configuration to optimize LLaMA 3 models for AMD NPU.

## ✅ Supported Models and Configs
| Model Name (Hugging Face)                         | Config File Name                  |
|:--------------------------------------------------|:----------------------------------|
| `meta-llama/Llama-3.1-8B-Instruct`                | `Llama-3.1-8B-Instruct_quark_vitisai_llm.json`  |

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
cd olive-recipes/meta-llama-Llama-3.1-8B-Instruct/VitisAI
pip install --force-reinstall -r requirements_vitisai_llm.txt
```

 

#### **Install PyTorch**

Make sure to install the correct version of PyTorch before running quantization:

**For AMD GPUs (ROCm):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

python -c "import torch; print(torch.cuda.is_available())" # Must return `True`
```

**For NVIDIA GPUs (CUDA):**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(torch.cuda.is_available())" # Must return `True`
```
#### **Generate optimized LLM model for VitisAI NPU**
Follow the above setup instructions, then run the below command to generate the optimized LLM model for VitisAI EP

```bash
# Llama-3.2-1B-Instruct
olive run --config Llama-3.2-1B-Instruct_quark_vitisai_llm.json

# Llama-3.2-3B-Instruct
olive run --config Llama-3.2-3B-Instruct_quark_vitisai_llm.json

# Meta-Llama-3-8B
olive run --config Meta-Llama-3-8B_quark_vitisai_llm.json

# Meta-Llama-3.1-8B
olive run --config Meta-Llama-3.1-8B_quark_vitisai_llm.json
```

✅ Optimized model saved in: `models/Meta-Llama-3-8B-vai/`

> **Note:** Output model is saved in `output_dir` mentioned in the json files.
