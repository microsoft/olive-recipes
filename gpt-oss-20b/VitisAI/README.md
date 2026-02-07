# Model Optimization and Quantization for AMD NPU

This folder contains sample Olive configuration to optimize Phi-4 models for AMD NPU.

## ✅ Supported Models and Configs

| Model Name (Hugging Face)                          | Config File Name                  |
|:---------------------------------------------------|:----------------------------------|
| `microsoft/gpt-oss-20b`                            | `gpt-oss-20b_quark_vitisai_llm.json`  |

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
cd olive-recipes/gpt-oss-20b/VitisAI
pip install --force-reinstall -r requirements_vitisai_llm.txt
```

#### **Generate optimized LLM model for VitisAI NPU**
GPT-OSS models are pre-quantized ONNX models that only need NPU optimization (no Quark quantization step).

Follow the above setup instructions, then run the below command to generate the optimized LLM model for VitisAI EP. 

1. Download the pre-quantized model:
```bash
hf download onnxruntime/gpt-oss-20b-onnx --include "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*" --local-dir ./models/gpt-oss-20b-onnx
```

2. Run the Olive recipe:
```bash
# Phi-4-mini-instruct
olive run --config gpt-oss-20b_quark_vitisai_llm.json
```

✅ Optimized model saved in: `models/gpt-oss-20b-vai/`
> **Note:** Output model is saved in `output_dir` mentioned in the json files.
