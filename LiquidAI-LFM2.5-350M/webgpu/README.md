# LiquidAI-LFM2.5-350M — WebGPU optimization

## Recipes

### `_webgpu_int4.json` — INT4 weights
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table stays FP16.

```
olive run --config LiquidAI-LFM2.5-350M_webgpu_int4.json
```

### `_webgpu_fp16_int4.json` — FP16 embedding + INT4 weights
INT4 weights via RTN, with the LM head excluded so both it and the embedding
table stay FP16. Larger than `_webgpu_int4.json`, and less accurate (see below).

```
olive run --config LiquidAI-LFM2.5-350M_webgpu_fp16_int4.json
```

## Measured quality

KL divergence of the next-token distribution from the FP32 model on wikitext-2 (`KLD`, lower is
better) and how often the most likely next token matches FP32 (`same top`), next to LiquidAI's GGUFs
on llama.cpp's CPU and Metal backends. ONNX rows: the WebGPU EP on an Apple M3 Ultra (Metal). `±` is
the standard error over the 64 scored chunks; sizes count the decoder. Method, scripts and the
measurements behind the notes: [eval](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md).

| recipe | size | KLD | same top |
| --- | --- | --- | --- |
| `_webgpu_int4.json` | 0.38 GiB | 0.491 ± 0.012 | 67.3% |
| `_webgpu_fp16_int4.json` | 0.42 GiB | 0.979 ± 0.022 | 55.2% |
| llama.cpp Q4_K_M, CPU | 0.21 GiB | 0.481 ± 0.010 | 67.4% |
| llama.cpp Q4_K_M, Metal | 0.21 GiB | 0.465 ± 0.010 | 68.0% |
| llama.cpp Q8_0, CPU | 0.35 GiB | 0.00843 ± 0.00018 | 95.3% |
| llama.cpp Q8_0, Metal | 0.35 GiB | 0.00370 ± 0.00009 | 96.7% |

- INT4 has 2% ± 1% more KLD than Q4_K_M on llama.cpp CPU and 6% ± 2% more than on Metal (paired on
  the same tokens), at 1.8x its size
  ([why](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md#size)).
- [microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814) (open) fits
  `k_quant`'s scale to the zero point it stores. With it, the CPU INT4 recipe scores 0.468 instead
  of 0.492: 3% ± 2% less KLD than Q4_K_M on llama.cpp CPU
  ([details](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md#int4-against-q4_k_m-k_quants-zero-point)).
  This recipe uses the same quantizer.
- Use `_webgpu_int4.json`: `_webgpu_fp16_int4.json` is larger and has 2.0x its KLD. Its FP16 LM head
  buys nothing; its symmetric RTN rounding and its 4-bit sensitive layers cost the difference
  ([measured](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md#fp16_int4-against-int4-on-webgpu)).
- Measured on Metal; D3D12 and Vulkan were not measured.

## Setup

Python 3.11+ is required: onnxruntime-genai stopped publishing cp310 wheels at 0.12.

```
pip install git+https://github.com/microsoft/olive.git
pip install -r requirements.txt
```

These recipes need Olive from `main` together with `onnxruntime-genai>=0.16.0`.
The released `olive-ai` package cannot drive genai 0.15+ (its ModelBuilder pass
skips the `check_extra_options` step that `create_model` now requires), and Olive
`main` imports the `onnxruntime_genai.models.loaders` package that only ships from
genai 0.16.0. LFM2 support itself landed in genai 0.14.0.
