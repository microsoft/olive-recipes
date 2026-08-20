# Qwen3-30B-A3B KQuant + Mobius (CUDA)

This recipe exports
[`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B)
as an ONNX Runtime GenAI text-generation package for CUDA.

The workflow applies Olive's PyTorch-side `KQuant` pass to the dense decoder
linears and fused MoE expert weights, then uses `MobiusBuilder` to export a
decoder-only model and its ORT GenAI configuration.

## Quantization

The recipe uses symmetric 4-bit weights with group size 128:

```json
{
    "bits": 4,
    "group_size": 128,
    "sym": true,
    "moe": true,
    "embeds": false,
    "lm_head": false
}
```

The 128 routed experts in every MoE layer are quantized independently. Routers,
token embeddings, normalization layers, and the language-model head remain
floating point. Mobius exports floating-point tensors and activations at fp16
precision.

## Setup and export

Run from the `Qwen-Qwen3-30B-A3B/` directory:

```bash
pip install -r cuda/requirements.txt
olive run --config cuda/kquant_fp16/config.json
```

The output is written to `cuda/kquant_fp16/models/`.

## Inference

Use the ORT GenAI text-generation example:

```bash
python /path/to/onnxruntime-genai/examples/python/model-generate.py \
  -m cuda/kquant_fp16/models \
  -e cuda \
  -pr "What is the capital of France?" \
  --non_interactive
```

This is a decoder-only package, so ORT GenAI uses its text-only runtime path;
no vision encoder, image processor, or multimodal pipeline is involved.

## Validation status

The complete workflow was validated on an NVIDIA A100 80 GB GPU with ONNX
Runtime 1.30.0 and ONNX Runtime GenAI 0.16.0-dev:

- KQuant produced a 16.7 GB quantized checkpoint.
- Mobius exported a 16.7 GB `model.onnx.data` file and the ORT GenAI package.
- ORT GenAI loaded the package in approximately 53-58 seconds.
- CUDA greedy generation produced coherent text, valid Python code, and the
  correct answer `391` for `17 * 23`.
- Qwen3 reasoning token metadata resolved correctly (`bor=151667`,
  `eor=151668`).

Observed generation throughput ranged from 61 to 182 output tokens per second
for the short smoke-test prompts after model load. This is a functional
validation result, not a controlled performance benchmark.

## References

- [Olive KQuant](https://github.com/microsoft/Olive/blob/main/olive/passes/pytorch/kquant.py)
- [Mobius](https://github.com/onnxruntime/mobius)
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)
