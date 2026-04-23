# Gemma 4 E2B (google/gemma-4-E2B-it)

Olive recipes for building and quantizing the Gemma 4 E2B multimodal model
using the [MobiusModelBuilder](https://github.com/microsoft/Olive) pass.

## Prerequisites

```bash
pip install olive-ai[gpu] mobius-ai
```

## Recipes

### FP32 CPU

Build the model in fp32 for CPU inference:

```bash
olive run --config gemma4_fp32_cpu.json
```

### INT4 CUDA

Build in fp16, then quantize to INT4 with block-wise RTN for CUDA:

```bash
olive run --config gemma4_int4_cuda.json
```

Output: 4 quantized ONNX components (decoder, vision_encoder, audio_encoder, embedding) totaling ~2.8GB.

## Model info

- **Architecture**: Gemma 4 any-to-any multimodal (vision + audio + text)
- **Components**: decoder, vision_encoder, audio_encoder, embedding
- **HuggingFace**: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
