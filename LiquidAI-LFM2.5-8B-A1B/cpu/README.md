# LiquidAI-LFM2.5-8B-A1B — CPU optimization

LFM2.5-8B-A1B is a sparse mixture-of-experts model: 8.3B total parameters, 1.5B active per
token. It keeps the LFM2 conv/attention stack, but only the first 2 of its 24 layers use a
dense SwiGLU MLP — the other 22 route each token to 4 of 32 experts.

The GenAI model builder exports those 22 layers as one fused `QMoE` op each, so
`algo_config` and `matmul_mixed_precision` reach only the dense MatMuls; the expert weights
are quantized block-wise by the builder itself through `qmoe_block_size`. The router MatMul
is always left in floating point so quantization rounding cannot change which experts a
token is sent to.

## Recipes

### `_cpu_int8.json` — closest to the original
Symmetric INT8 weights and INT8 experts, straight from the model builder. Use this when
10 GiB is acceptable: it costs about 60% more disk than INT4 and, on x86, decodes about 10%
slower, and it is far closer to the original model.

```
olive run --config LiquidAI-LFM2.5-8B-A1B_cpu_int8.json
```

Unlike the dense LFM2.5 INT8 recipes, this one has no `rtn` pass in front of the model
builder: Olive's `rtn` pass cannot walk an MoE model (it looks for `model.layers.N.mlp`,
which `Lfm2MoeForCausalLM` does not have), and the builder's own INT8 path already
quantizes the experts and the LM head. The embedding table stays in floating point: the
builder only packs the embedding `Gather` at 4 bits, which is also why the export logs a
harmless `Gather only supports 4 bits quantization` error.

### `_cpu_int4.json` — smallest
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table is left unquantized.
The experts are INT4 with a block size of 32.

```
olive run --config LiquidAI-LFM2.5-8B-A1B_cpu_int4.json
```

## Measured quality and speed

Logits compared against the Hugging Face FP32 model over 819 scored positions (8 chat
prompts: prose, code, arithmetic, history, documentation, data structures, translation,
systems). `top-1` is agreement with the FP32 argmax; `KL` is KL(fp32 || onnx) averaged per
position. Decode is greedy generation through onnxruntime-genai. Measured on a 30-core
Xeon 8358 with an ONNX Runtime 1.31 nightly, which has the fast `QMoE` path described below.

| recipe       | size     | top-1 | KL     | decode |
| ------------ | -------- | ----- | ------ | ------ |
| int8         |  9.9 GiB | 0.970 | 0.0086 | 48 t/s |
| int4         |  6.1 GiB | 0.891 | 0.0852 | 55 t/s |
| (plain int4) |  5.1 GiB | 0.858 | 0.1309 |        |

ONNX Runtime 1.30.0 scores the same files INT8 0.967 / 0.0073 and INT4 0.888 / 0.0918.
Top-1 is within noise for both (paired McNemar p = 0.84 and 0.58), as is INT8's KL; INT4's
KL is slightly better on the nightly (paired bootstrap CI excludes zero).
On an Apple M3 Ultra the same recipes score INT8 0.976 / 0.0083 at 83 t/s and INT4
0.894 / 0.0899 at 62 t/s, so on Arm64 INT8 is also the faster of the two.

The INT4 recipe's `matmul_mixed_precision` settings earn their extra 1 GiB: against plain
INT4 they improve top-1 by 3.3 points (paired McNemar p = 0.005) and KL by 35% (paired
bootstrap CI excludes zero). `qmoe_block_size: 32` is likewise measured, not assumed — see
the CUDA README for the block-size comparison.

Against llama.cpp on wikitext-2 (KL divergence from the FP32 model over 16,320 tokens;
LiquidAI's GGUFs scored with `llama-perplexity --kl-divergence` on its Metal and CPU
backends, ONNX on an Apple M3 Ultra with ONNX Runtime 1.30.0): INT8 scores 0.024 against
Q8_0's 0.020-0.029, and INT4 0.207 against Q4_K_M's 0.175-0.179, at 6.1 GiB against 4.8 GiB.
INT4 differs from Q4_K_M in two ways: the model builder quantizes the experts, which hold
most of the weights, with symmetric block-wise rounding (no zero point) where Q4_K fits a
scale and a minimum per block, and the dense MatMuls go through ONNX Runtime's `k_quant`,
whose rounding
[microsoft/onnxruntime#32814](https://github.com/microsoft/onnxruntime/pull/32814) fixes. On
this MoE model even llama.cpp's BF16 GGUF sits at 0.009 from the PyTorch FP32 model: small
numeric differences flip the router's expert choices.

## Decode speed needs a current ONNX Runtime

The CPU `QMoE` kernel in ONNX Runtime 1.30.0 dequantizes every expert back to fp32 on each
call, which leaves decode at about 1 tok/s for INT4 and 5 tok/s for INT8 on the Xeon above.
[microsoft/onnxruntime#32644](https://github.com/microsoft/onnxruntime/pull/32644) runs the
block-wise experts directly on the MLAS QNBit GEMM (`MatMulNBits`) kernels instead, which is
where the 48-55 tok/s above comes from. It landed after 1.30.0, so until 1.31 is released
install an ONNX Runtime nightly after the requirements:

```
pip install --pre --force-reinstall --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime
```

Both precisions pick the fast path automatically; `ORT_QMOE_CPU_QNBIT_GEMM=0` disables it,
which is a quick way to confirm which path a build is taking.

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
genai 0.16.0. LFM2-MoE support itself comes from
[microsoft/onnxruntime-genai#2575](https://github.com/microsoft/onnxruntime-genai/pull/2575),
which landed after 0.16.0 and changes the runtime as well as the model builder. Until a
release includes it, build onnxruntime-genai from `main` and install that wheel after the
requirements with `pip install --no-deps --force-reinstall`: a `main` build is versioned
`0.16.0.dev0`, which sorts below `0.16.0`, so installing the requirements afterwards would
replace it with the release.
