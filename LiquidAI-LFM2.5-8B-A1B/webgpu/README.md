# LiquidAI-LFM2.5-8B-A1B — WebGPU optimization

LFM2.5-8B-A1B is a sparse mixture-of-experts model: 8.3B total parameters, 1.5B active per
token. It keeps the LFM2 conv/attention stack, but only the first 2 of its 24 layers use a
dense SwiGLU MLP — the other 22 route each token to 4 of 32 experts.

The GenAI model builder exports those 22 layers as one fused `QMoE` op each, so
`algo_config` and `matmul_mixed_precision` reach only the dense MatMuls; the expert weights
are quantized block-wise by the builder itself through `qmoe_block_size`. The router MatMul
is always left in floating point so quantization rounding cannot change which experts a
token is sent to.

## Recipes

### `_webgpu_int4.json` — Q4_K_M equivalent
INT4 weights via k_quant, with `matmul_mixed_precision` keeping the sensitive
layers and the LM head at INT8. The embedding table stays FP16.
The experts are INT4 with a block size of 32.

```
olive run --config LiquidAI-LFM2.5-8B-A1B_webgpu_int4.json
```

### `_webgpu_int8.json` — closest to the original
Symmetric INT8 weights and INT8 experts, straight from the model builder. The embedding
table stays FP16. Use this when the GPU can hold 9 GiB: it matches the FP16 export in our
measurements at 55% of its size.

```
olive run --config LiquidAI-LFM2.5-8B-A1B_webgpu_int8.json
```

## Measured quality and speed

Logits compared against the Hugging Face FP32 model over 819 scored positions (8 chat
prompts: prose, code, arithmetic, history, documentation, data structures, translation,
systems). `top-1` is agreement with the FP32 argmax; `KL` is KL(fp32 || onnx) averaged per
position. Decode is greedy generation through onnxruntime-genai. Measured on an Apple M3
Ultra (Metal) with the `onnxruntime-ep-webgpu` 0.4.0 plugin.

| recipe      | size     | top-1 | KL     | decode |
| ----------- | -------- | ----- | ------ | ------ |
| (fp16)      | 16.3 GiB | 0.930 | 0.0324 |  5 t/s |
| int8        |  8.9 GiB | 0.926 | 0.0349 | 74 t/s |
| int4        |  5.1 GiB | 0.876 | 0.1027 | 78 t/s |
| (fp16_int4) |  5.3 GiB | 0.867 | 0.1465 | 74 t/s |

INT8 is statistically indistinguishable from FP16 here: paired McNemar on top-1 gives
p = 0.64, and the paired bootstrap CI for the KL difference spans zero. The FP16 row is a
reference only: unquantized experts run through WebGPU's `MoE` op, which is far slower than
the `QMoE` path both recipes use.

WebGPU runs the model in FP16, and its FP16 ceiling sits below the CUDA one (0.959 / 0.0224,
see the CUDA README). Since INT8 reaches that ceiling, the remaining gap to CUDA and CPU
comes from the FP16 kernels rather than from quantization.

The `fp16_int4` layout of the dense LFM2.5 WebGPU recipes (RTN INT4 with the LM head left
in FP16) was measured too and is dominated: it is larger than `_webgpu_int4.json`, no
better on top-1 (p = 0.53), and worse on KL (paired bootstrap CI excludes zero). With 22 of
24 MLPs in `QMoE`, the k_quant and `matmul_mixed_precision` treatment of the remaining dense
MatMuls matters more than an FP16 LM head, so it is deliberately not a recipe here.

## Running on WebGPU

The WebGPU execution provider ships as a plugin package:

```
pip install onnxruntime-ep-webgpu
```

onnxruntime-genai's `examples/python` scripts register it automatically when it is
installed. In your own code, register it before loading the model:

```python
import onnxruntime_ep_webgpu as webgpu
import onnxruntime_genai as og

og.register_execution_provider_library(webgpu.get_ep_name(), webgpu.get_library_path())
model = og.Model("model")
```

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
