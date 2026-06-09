# Package Qwen3.5-0.8B into an ONNX Runtime ortpackage

This recipe walks through the end-to-end flow for taking a Hugging Face VLM
(`Qwen/Qwen3.5-0.8B`), exporting it to ONNX with Olive's Mobius builder for
both CPU and GPU, and bundling the two exports into a single
`.ortpackage` that the ONNX Runtime GenAI runtime can load and switch
between at inference time.

## Prerequisites

`pip install -r requirements.txt`

## Step 1 — Export the GPU variant (fp16, default)

```bash
olive capture-onnx-graph -m Qwen/Qwen3.5-0.8B --use_mobius_builder -o gpu
```

This downloads the HF model, runs the Mobius ONNX builder, and writes a
multi-component VLM source under `gpu/`:


## Step 2 — Export the CPU variant (fp32)

```bash
olive capture-onnx-graph -m Qwen/Qwen3.5-0.8B --use_mobius_builder -o cpu --precision fp32
```


## Step 3 — Combine both variants into one `.ortpackage`

```bash
olive generate-model-package -s cpu -s gpu -o cpu_gpu
```

Multiple `-s/--source` flags can be passed, producing `cpu_gpu.ortpackage/` here.


## Step 4 — Load and run the package with onnxruntime-genai

```python
import onnxruntime_genai as og

model = og.Model("cpu_gpu.ortpackage", ep="cpu")
tokenizer = og.Tokenizer(model)

params = og.GeneratorParams(model)
params.set_search_options(max_length=64, do_sample=False)

generator = og.Generator(model, params)
input_tokens = tokenizer.encode("Hello!")
generator.append_tokens(input_tokens)
while not generator.is_done():
    generator.generate_next_token()

full = list(generator.get_sequence(0))
new_tokens = full[len(input_tokens):]
print(tokenizer.decode(new_tokens))
```

To run on GPU instead, change `ep="cpu"` to `ep="cuda"` and ensure the
matching ORT GPU EP package is installed.
