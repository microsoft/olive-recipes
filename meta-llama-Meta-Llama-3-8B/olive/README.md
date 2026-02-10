# 2-bit Llama-3-8B Quantization

This recipe demonstrates how to use Olive to export a prequantized Llama-3-8b model from Huggingface to an optimized ONNX model and run inference using ONNX Runtime GenAI.

## Pre-requisites
Install Olive and other dependencies:

```bash
pip install -r requirements.txt
```

## Prepare the Checkpoint
We use the model [`ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ`](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ) from huggingface which is quantized using EfficientQAT to 2-bit weights with group size of 128. The weights are stored in the `gptq_v2` format so we need to convert them into Olive's quantization format first.

```bash
python prepare_ckpt.py -m ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ -o models/llama-3-8b-pt
```

## Run Graph Capture using Olive
To export the model to ONNX format, run the following command:

```bash
olive run --config config.json -m models/llama-3-8b-pt -o models/llama-3-8b-onnx
```

## Inference with ONNX Runtime GenAI
You can run inference using the exported ONNX model as follows:
```python
import json
import time

import onnxruntime_genai as og

config = og.Config("models/llama-3-8b-onnx")
config.overlay(json.dumps({
    "model": {
        "decoder": {
            "session_options": {
                "mlas.use_lut_gemm": "1"
            }
        }
    }
}))
model = og.Model(config)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

prompt = "Model quantization is"
# prompt = "the sky is blue because of the scattering"
# prompt = "once upon a time in a land far away,"
input_tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=128)
generator = og.Generator(model, params)

generator.append_tokens(input_tokens)
start = time.time()
first_token_time = None
num_tokens = 0
while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    if first_token_time is None:
        first_token_time = time.time()
    print(tokenizer_stream.decode(new_token), end="", flush=True)
    num_tokens += 1

end = time.time()
print(f"\nTime to first token: {(first_token_time - start) * 1000:.2f} ms")
decoding_time = end - first_token_time
print(f"Tokens {num_tokens - 1}, Tokens/sec: {(num_tokens - 1) / decoding_time:.2f}")
```
