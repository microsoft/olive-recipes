# LiquidAI-LFM2.5-8B-A1B — CUDA optimization

LFM2.5-8B-A1B is a sparse mixture-of-experts model: 8.3B total parameters, 1.5B active per
token. It keeps the LFM2 conv/attention stack, but only the first 2 of its 24 layers use a
dense SwiGLU MLP — the other 22 route each token to 4 of 32 experts.

The GenAI model builder exports those 22 layers as one fused `QMoE` op each, so
`algo_config` and `matmul_mixed_precision` reach only the dense MatMuls; the expert weights
are quantized block-wise by the builder itself through `qmoe_block_size`. The router MatMul
is always left in floating point so quantization rounding cannot change which experts a
token is sent to.

## Recipes

### `_cuda_int8.json` — closest to the original
Symmetric INT8 weights and INT8 experts, straight from the model builder. This is the
recommended recipe when the model fits: it is indistinguishable from FP16 in our
measurements, at 55% of FP16's size and 1.6x its decode speed.

```
olive run --config LiquidAI-LFM2.5-8B-A1B_cuda_int8.json
```

### `_cuda_int4.json` — smallest
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table is left unquantized.
The experts are INT4 with a block size of 32. Use this when 5 GiB matters more than the
accuracy gap to INT8.

```
olive run --config LiquidAI-LFM2.5-8B-A1B_cuda_int4.json
```

## Measured quality

Logits compared against the Hugging Face FP32 model over 819 scored positions (8 chat
prompts: prose, code, arithmetic, history, documentation, data structures, translation,
systems), on an A10. `top-1` is agreement with the FP32 argmax; `KL` is KL(fp32 || onnx)
averaged per position. Decode is greedy generation through onnxruntime-genai, with the
`onnxruntime-ep-cuda12` plugin on an ONNX Runtime 1.31 nightly.

| recipe   | size     | top-1 | KL     | decode  |
| -------- | -------- | ----- | ------ | ------- |
| (fp16)   | 16.3 GiB | 0.959 | 0.0224 | 115 t/s |
| int8     |  8.9 GiB | 0.955 | 0.0175 | 180 t/s |
| int4     |  5.1 GiB | 0.880 | 0.0983 | 219 t/s |

INT8 is statistically indistinguishable from FP16 here: paired McNemar on top-1 gives
p = 0.65, and the paired bootstrap CI for the KL difference spans zero.

Two settings were chosen by measurement rather than by analogy with the dense LFM2.5
recipes, so leave them alone unless you re-measure:

- `qmoe_block_size: 32` is the finest block size CUDA supports, and it matters. At 128 the
  same INT4 recipe loses 40% on KL (0.098 -> 0.138) and 2.2 points of perplexity to save
  0.34 GiB.
- The INT4 recipe's `matmul_mixed_precision` is worth its 0.5 GiB. Against plain INT4 it
  improves KL on both CPU and CUDA (paired bootstrap CI excludes zero) and top-1 by 2-3
  points. Dropping only the `last_matmul:int8` half is measurably worse on KL.

`moe_quant_type=int8` (INT8 experts with INT4 dense weights) was also measured and is
dominated: 8.6 GiB for top-1 0.926, worse than plain INT8 on every metric at nearly the
same size. It is deliberately not a recipe here.

Against llama.cpp on wikitext-2 (KL divergence from the FP32 model over 16,320 tokens;
LiquidAI's GGUFs scored with `llama-perplexity --kl-divergence` on its Metal and CPU
backends, ONNX on the A10): INT8 scores 0.027 against Q8_0's 0.020-0.029, and INT4 0.202
against Q4_K_M's 0.175-0.179, at 5.1 GiB against 4.8 GiB. INT4 differs from Q4_K_M in two
ways: the model builder quantizes the experts, which hold most of the weights, with
symmetric block-wise rounding (no zero point) where Q4_K fits a scale and a minimum per
block, and the dense MatMuls go through ONNX Runtime's `k_quant`, whose rounding
[microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814) fixes. On
this MoE model even llama.cpp's BF16 GGUF sits at 0.009 from the PyTorch FP32 model: small
numeric differences flip the router's expert choices.

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
genai 0.16.0. LFM2-MoE support itself comes from
[microsoft/onnxruntime-genai#2575](https://github.com/microsoft/onnxruntime-genai/pull/2575),
which landed after 0.16.0 and changes the runtime as well as the model builder. Until a
release includes it, build onnxruntime-genai from `main` and install that wheel after the
requirements with `pip install --no-deps --force-reinstall`: a `main` build is versioned
`0.16.0.dev0`, which sorts below `0.16.0`, so installing the requirements afterwards would
replace it with the release.
