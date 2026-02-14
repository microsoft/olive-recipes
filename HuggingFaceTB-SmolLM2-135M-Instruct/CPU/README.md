# SmolLM2-135M-Instruct Optimization Recipe for CPU

This folder contains the optimization recipe for the [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) model targeting CPU execution. The model is optimized to **INT4** precision using Microsoft Olive and ONNX Runtime GenAI's ModelBuilder.

## 📊 Recipe Details

| Property | Details |
| :--- | :--- |
| **Model Name** | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| **Architecture** | SmolLM2 (Llama-based) |
| **Target Device** | CPU |
| **Precision** | INT4 |
| **Execution Provider** | `CPUExecutionProvider` |
| **Optimization Tool** | Microsoft Olive (ModelBuilder) |

## 🛠️ Prerequisites

Before running the optimization, ensure you have the required dependencies installed.

```bash
pip install -r requirements.txt
```

## 🚀 How to Run Optimization

Navigate to this directory and run the following command to optimize the model:

```bash
python -m olive run --config olive_config.json
```

This will download the model, apply INT4 quantization, and save the optimized ONNX model in the `models/smollm_manual` directory.

## 🤖 How to Run Inference

Once the model is optimized, you can use `onnxruntime-genai` to run inference locally.

**Example Python Snippet:**

```python
import onnxruntime_genai as og

model = og.Model("models/smollm_manual")
tokenizer = og.Tokenizer(model)

prompt = "<|im_start|>user\nExplain quantum physics in one sentence.<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=100)
params.input_ids = tokens

generator = og.Generator(model, params)

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()
    print(tokenizer.decode(generator.get_next_tokens()), end='', flush=True)
```