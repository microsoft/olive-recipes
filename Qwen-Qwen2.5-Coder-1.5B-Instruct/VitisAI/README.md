# Model Optimization and Quantization for AMD NPU
This folder contains sample Olive configuration to optimize Qwen models for AMD NPU.

## ✅ Supported Models and Configs

| Model Name (Hugging Face)          | Config File Name                |
| :--------------------------------- | :------------------------------ |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct`       | `Qwen2.5-Coder-1.5B-Instruct_quark_vitisai_llm.json` |

## **Run the Quantization Config**

### **Quark quantization**

For LLMs - follow the below commands to generate the optimized model for VitisAI Execution Provider.

**Platform Support:**
- ✅ **Windows with CUDA** - Supported
- ✅ **Windows with CPU** - Supported
- ⏳ **Planned for future release:** Linux with ROCm, Linux with CUDA, Windows with ROCm

For more details about quark, see the [Quark Documentation](https://quark.docs.amd.com/latest/)

#### **Create a Python 3.12 conda environment and run the below commands**
```bash
conda create -n olive python=3.12
conda activate olive
```

#### **Install Olive**

**Option 1: Install from PyPI**
```bash
pip install olive-ai[auto-opt]
pip install transformers onnxruntime-genai
```

**Option 2: Install from source**
```bash
git clone https://github.com/microsoft/Olive.git
cd Olive
pip install -e .
pip install -r requirements.txt
```

#### **Install VitisAI LLM dependencies**

```bash
cd olive-recipes/Qwen-Qwen2.5-Coder-1.5B-Instruct/VitisAI
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

**For CPU-only (Windows):**
```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"  # Should print 2.7.0+cpu
```

#### **Generate optimized LLM model for VitisAI NPU**
Follow the above setup instructions, then run the below command to generate the optimized LLM model for VitisAI EP

```bash
# Qwen2.5-Coder-1.5B-Instruct
olive run --config Qwen2.5-Coder-1.5B-Instruct_quark_vitisai_llm.json
```

✅ Optimized model saved in: `models/Qwen2.5-Coder-1.5B-Instruct-vai/`

> **Note:** Output model is saved in `output_dir` mentioned in the json files.
