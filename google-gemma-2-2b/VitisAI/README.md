# Model Optimization and Quantization for AMD NPU

This folder contains sample Olive configurations to optimize DeepSeek models for AMD NPU.

## ✅ Supported Models and Configs

| Model Name                                               | Config File Name                                      |
|:---------------------------------------------------------|:------------------------------------------------------|
| `google/gemma-2-2b`               | `gemma-2-2b_quark_vitisai_llm.json` |

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
cd olive-recipes/google-gemma-2-2b/VitisAI
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
# gemma-2-2b
olive run --config gemma-2-2b_quark_vitisai_llm.json
```

✅ Optimized model saved in: `models/gemma-2-2b-vai/`
> **Note:** Output model is saved in `output_dir` mentioned in the json files.
