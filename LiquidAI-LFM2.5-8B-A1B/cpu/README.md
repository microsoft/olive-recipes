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
10 GiB is acceptable: it costs about 60% more disk than INT4 and decodes about 10% slower,
and it is far closer to the original model.

```
olive run --config LiquidAI-LFM2.5-8B-A1B_cpu_int8.json
```

Unlike the dense LFM2.5 INT8 recipes, this one has no `rtn` pass in front of the model
builder: Olive's `rtn` pass cannot walk an MoE model (it looks for `model.layers.N.mlp`,
which `Lfm2MoeForCausalLM` does not have), and the builder's own INT8 path already
quantizes the experts, the embeddings and the LM head.

### `_cpu_int4.json` — Q4_K_M equivalent
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
Xeon 8358.

| recipe       | size     | top-1 | KL     | decode |
| ------------ | -------- | ----- | ------ | ------ |
| int8         |  9.9 GiB | 0.96  | 0.009  | 45 t/s |
| int4         |  6.1 GiB | 0.891 | 0.0852 | 51 t/s |
| (plain int4) |  5.1 GiB | 0.858 | 0.1309 |        |

The INT8 row is rounded because it moves slightly with the runtime: the INT8 `QMoE` kernel
switched to int8 activations in the build referenced below, which costs a little accuracy
for a large speed gain (the same model scores 0.967 / 0.0073 on a runtime predating it).
The INT4 rows are identical on both builds.

The INT4 recipe's `matmul_mixed_precision` settings earn their extra 1 GiB: against plain
INT4 they improve top-1 by 3.3 points (paired McNemar p = 0.005) and KL by 35% (paired
bootstrap CI excludes zero). `qmoe_block_size: 32` is likewise measured, not assumed — see
the CUDA README for the block-size comparison.

## Decode speed needs a current ONNX Runtime

The CPU `QMoE` kernel used to dequantize every expert back to fp32 on each call, which left
decode at about 3 tok/s for this model at either precision. ONNX Runtime builds that include
[microsoft/onnxruntime#32644](https://github.com/microsoft/onnxruntime/pull/32644) run the
block-wise experts directly on the MLAS QNBit GEMM (`MatMulNBits`) kernels instead, which is
where the 45-51 tok/s above comes from. Both precisions pick the fast path automatically;
`ORT_QMOE_CPU_QNBIT_GEMM=0` disables it, which is a quick way to confirm which path a build
is taking.

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
[microsoft/onnxruntime-genai#2575](https://github.com/microsoft/onnxruntime-genai/pull/2575);
until that is released, build the wheel from a branch that contains it.
