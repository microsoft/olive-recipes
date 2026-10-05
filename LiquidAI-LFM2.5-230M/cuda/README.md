# LiquidAI-LFM2.5-230M — CUDA optimization

## Recipes

### `_cuda_int4.json` — INT4 weights
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table is left unquantized.

```
olive run --config LiquidAI-LFM2.5-230M_cuda_int4.json
```

### `_cuda_int8.json` — INT8 weights
Symmetric INT8 weights throughout, with the embedding table and LM head also at INT8 via RTN.

```
olive run --config LiquidAI-LFM2.5-230M_cuda_int8.json
```

## Measured quality

KL divergence of the next-token distribution from the FP32 model on wikitext-2 (`KLD`, lower is
better) and how often the most likely next token matches FP32 (`same top`), next to LiquidAI's GGUFs
on llama.cpp's CPU and Metal backends. ONNX rows: the CUDA EP on an NVIDIA A10. `±` is the standard
error over the 64 scored chunks; sizes count the decoder. Method, scripts and the measurements
behind the notes: [eval](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md).

| recipe | size | KLD | same top |
| --- | --- | --- | --- |
| `_cuda_int4.json` | 0.31 GiB | 0.110 ± 0.002 | 80.5% |
| `_cuda_int8.json` | 0.31 GiB | 0.00170 ± 0.00006 | 97.2% |
| llama.cpp Q4_K_M, CPU | 0.14 GiB | 0.0988 ± 0.0017 | 81.1% |
| llama.cpp Q4_K_M, Metal | 0.14 GiB | 0.0922 ± 0.0015 | 81.7% |
| llama.cpp Q8_0, CPU | 0.23 GiB | 0.00167 ± 0.00003 | 97.3% |
| llama.cpp Q8_0, Metal | 0.23 GiB | 0.000744 ± 0.000012 | 98.2% |

- INT4 has 11% ± 2% more KLD than Q4_K_M on llama.cpp CPU and 19% ± 2% more than on Metal (paired on
  the same tokens), at 2.2x its size
  ([why](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md#size)).
- [microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814) (open) fits
  `k_quant`'s scale to the zero point it stores. With it, the CPU INT4 recipe scores 0.0950 instead
  of 0.114: 4% ± 2% less KLD than Q4_K_M on llama.cpp CPU
  ([details](../../LiquidAI-LFM2.5-1.2B-Instruct/eval/README.md#int4-against-q4_k_m-k_quants-zero-point)).
  This recipe uses the same quantizer.
- INT8 has 2% ± 3% more KLD than Q8_0 on llama.cpp CPU and 2.3x Metal's.

## Setup

Python 3.11+ is required: onnxruntime-genai stopped publishing cp310 wheels at 0.12.

```
pip install git+https://github.com/microsoft/olive.git
pip install -r requirements.txt
```

These recipes need Olive from `main` together with `onnxruntime-genai-cuda>=0.16.0`.
The released `olive-ai` package cannot drive genai 0.15+ (its ModelBuilder pass
skips the `check_extra_options` step that `create_model` now requires), and Olive
`main` imports the `onnxruntime_genai.models.loaders` package that only ships from
genai 0.16.0. LFM2 support itself landed in genai 0.14.0.

The current `onnxruntime-genai-cuda` and `onnxruntime-gpu` wheels on PyPI are CUDA 13 builds and need
NVIDIA driver 580 or newer (`nvidia-smi` shows the driver version). No CUDA 12 wheels of
`onnxruntime-genai-cuda` 0.16 or newer are published, so with an older driver either upgrade it or
[build onnxruntime-genai from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html)
against a CUDA 12 build of onnxruntime.
