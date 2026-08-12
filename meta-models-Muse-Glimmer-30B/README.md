# Muse Glimmer 30B

This CUDA recipe exports
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)
to a three-model ONNX Runtime GenAI package with Mobius, then quantizes it
to INT4 with Olive K-Quant.

Muse Glimmer is a 29.6B-parameter vision-language model with a dense text
decoder and a dedicated 1.8B-parameter perception encoder. It supports
interleaved text and images and emits text. The source checkpoint is
licensed under Apache 2.0.

## Pipeline

| Recipe | Pipeline | Device / EP | Output |
|---|---|---|---|
| `cuda/int4/config.json` | `MobiusBuilder(bf16)` -> `OnnxKQuantQuantization(bits=4, block_size=32)` | NVIDIA GPU / CUDAExecutionProvider | `cuda/int4/models` |

Mobius produces the complete multimodal package:

```text
cuda/int4/models/
├── decoder/model.onnx
├── embedding/model.onnx
├── vision_encoder/model.onnx
├── genai_config.json
├── processor_config.json
├── tokenizer.json
└── tokenizer_config.json
```

The BF16 export matches the checkpoint's native precision. K-Quant reduces
the weight footprint from roughly 60 GB to under 20 GB, making the package
practical on 24 GB and 32 GB CUDA systems. Quantization requires enough host
memory for the source checkpoint and intermediate ONNX model. Installing a
CUDA-matched CuPy package can substantially accelerate K-Quant.

The generated ORT GenAI configuration enables CUDA Graph capture for the
autoregressive decoder while leaving the one-shot vision encoder and embedding
sessions uncaptured.

## Prerequisites

- NVIDIA GPU with BF16 and INT4 support.
- CUDA 12.x and a compatible NVIDIA driver.
- Python 3.10 or newer.
- Approximately 80 GB of free disk space during conversion.
- Access to the public Hugging Face checkpoint.

```bash
pip install -r requirements.txt
pip install onnxruntime-genai-cuda

# Optional: accelerate K-Quant on CUDA 12.x.
pip install cupy-cuda12x
```

The requirements pin Mobius to the Muse Glimmer implementation validated
against the checkpoint. The pin can move to a released `mobius-onnx` version
after that support is published.

Native runtime loading requires
[microsoft/onnxruntime-genai#2397](https://github.com/microsoft/onnxruntime-genai/pull/2397)
until Muse Glimmer support reaches an ONNX Runtime GenAI release.

## Build

Run from this directory so Olive resolves the relative output path here:

```bash
olive run --config cuda/int4/config.json
```

The build downloads about 60 GB of BF16 weights. It exports the decoder,
vision encoder, and embedding mixer, generates ORT GenAI processor and
tokenizer assets, and quantizes eligible ONNX weights to Q4_K_M-style INT4.

## Inference

Text-only:

```bash
python inference.py --prompt "Explain why the sky is blue."
```

Image and text:

```bash
python inference.py \
  --image path/to/image.jpg \
  --prompt "Describe the image and identify any text."
```

Override `--model-path` when the generated package is stored elsewhere.
Generation is greedy by default for reproducible smoke testing.

## Validation

The Mobius implementation is covered by:

- full-size graph construction for all three components;
- tiny Hugging Face parity for text and the complete vision-to-decoder path;
- real-checkpoint BF16 CUDA prefill comparison;
- real-checkpoint BF16 CUDA 30-token greedy generation.

See [onnxruntime/mobius#475](https://github.com/onnxruntime/mobius/pull/475)
for implementation and validation details.

The complete recipe was also run on an NVIDIA H200:

- BF16 export and INT4 quantization completed in 1,436 seconds;
- the three-model ORT GenAI package occupied 18 GB;
- deterministic CUDA text generation produced a coherent Rayleigh-scattering answer;
- CUDA image generation correctly identified the test image as a Chinatown street
  scene with a traditional archway.

## References

- [Muse Glimmer model card](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [Mobius](https://github.com/onnxruntime/mobius)
- [Olive MobiusBuilder](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
- [Olive K-Quant pass](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/kquant_quantization.py)
