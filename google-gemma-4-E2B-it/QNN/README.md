# Gemma 4 E2B QNN (Snapdragon Hexagon NPU) recipe

> **Status: WORK IN PROGRESS / EXPLORATORY.** Starting-point recipe for
> running [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
> on the Qualcomm Hexagon NPU via the QNN ONNX Runtime execution
> provider (Snapdragon X / Copilot+ PC, Snapdragon 8 Gen 3+, etc.).
> The HTP EPContext compilation has *not* yet been validated on hardware
> — see [Limitations](#limitations).

## Two build paths

This directory ships **two** ways to produce the INT4 Gemma 4 E2B ONNX
model that the QNN AOT stages then compile to an HTP context binary:

| Path | Config | INT4 source | Cost |
|---|---|---|---|
| **GGUF-direct (recommended)** | `config_gguf.json` | reuse [`unsloth/gemma-4-E2B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) Q4_K_M weights | cheap — no calibration/RTN quant |
| HF-source | `config.json` | quantize fp32 weights with `OnnxKQuantQuantization` | expensive (RTN over 262 k-vocab; GPU / cupy recommended) |

The **GGUF-direct** path is the reason to use the unsloth GGUF: it skips
the expensive `OnnxKQuantQuantization` pass entirely by re-packing the
already-INT4 GGUF weights into `com.microsoft::MatMulNBits`, then runs
the identical QNN AOT stages. `mobius build-gguf` performs the GGUF →
ONNX conversion; the resulting model has been **verified end-to-end** to
produce correct text on ONNX Runtime CPU (see below).

### GGUF-direct: step 1 — build the INT4 ONNX with mobius

Run this once, on any machine (x64 or Windows ARM64), in the quantization
environment:

```bash
pip install "mobius-ai[gguf]"
mobius build-gguf "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_K_M.gguf" \
    --keep-quantized --ep qnn --dtype f16 --output model_int4
```

This emits `model_int4/model.onnx` — the Gemma 4 E2B text decoder. With
`--ep qnn` the mobius QNN capability now **decomposes** the fused opset-24
`Attention` op into scaled-dot-product primitives (Reshape / Transpose /
MatMul / Softmax / Add / Tile) and **inlines** `com.microsoft::MatMulNBits`
into a QDQ form (nibble-unpack + blocked `DequantizeLinear` + `MatMul`),
because the QNN HTP backend has no kernel for either fused op (they would
otherwise be forced onto CPU — see
[onnxruntime/onnxruntime-qnn#646](https://github.com/onnxruntime/onnxruntime-qnn/issues/646)).
`RMSNormalization` is kept (HTP runs it). Mixed Q6_K tensors in the Q4_K_M
preset are re-quantized to 4-bit / block-32 automatically.

Because mobius already lowers `MatMulNBits` → QDQ for `--ep qnn`, the Olive
`MatMulNBitsToQDQ` pass is a **no-op** on this model and is omitted from
`config_gguf.json`.

> **Static shapes for HTP.** The HTP backend requires *fully static*
> shapes. mobius `build-gguf` now accepts `--static-cache --max-seq-len N`
> (pre-allocated fixed-width KV buffers written in place via
> `TensorScatter`) to emit a static-shaped graph. **Caveat:** the Gemma-4
> decoder does not yet implement the `StaticCacheState` dispatch (it has
> KV-shared / sliding-window layers with dual RoPE), so
> `--static-cache` currently raises `TypeError` for gemma4. Adding that
> support is the main prerequisite for the QNN AOT path below — see
> [Limitations](#limitations).

> **Requires the mobius Gemma-4 GGUF fixes**
> ([onnxruntime/mobius PR](https://github.com/onnxruntime/mobius/pull/new/justinchuby/gemma4-gguf-qnn)):
> per-layer `feed_forward_length` → scalar + `use_double_wide_mlp`,
> per-layer-input tensor mapping, quantized-linear wiring, KV-shared
> standard-Attention shape inference, and the `gelu_pytorch_tanh`
> activation default (Gemma GGUFs omit the activation key; the old `silu`
> default produced garbage output). Without these, the build either fails
> to load in ORT or generates incoherent text.

Verified on Windows ARM64 (Snapdragon), ONNX Runtime 1.27 CPU EP, from
`model_int4/model.onnx`:

```
Q: What is the capital of France?   A: The capital of France is **Paris**.
Q: Name a primary color.            A: Red
```

The dequantized GGUF weights were confirmed to match the gated
`google/gemma-4-E2B-it` safetensors checkpoint (all projection / norm /
embedding correlations ≥ 0.98).

### GGUF-direct: step 2 — compile to a QNN HTP context binary

Point `config_gguf.json`'s `input_model.model_path` at the
`model_int4/model.onnx` produced above (and set the `qnn_system` Python
path), then run Olive from the quantization environment:

```bash
olive run --config config_gguf.json
```

`config_gguf.json` drops `MobiusBuilder`, `OnnxKQuantQuantization`, **and**
`MatMulNBitsToQDQ` (the mobius `--ep qnn` build already emits decomposed
Attention + QDQ INT4 weights) and runs only
`OnnxStaticQuantization → StaticLLM → EPContextBinaryGenerator`.

> **Known blocker (this recipe is not yet hardware-validated).** Two
> gaps remain before the AOT stages compile the full model to HTP:
>
> 1. **Full-model single-graph HTP compose fails on size.** The complete
>    35-layer / 4.65 B-param E2B graph returns `EP_FAIL: Failed to
>    compose Qnn graph`. Bisection shows small/medium subsets (up to a few
>    full-width layers, 262 k vocab, dual head_dim) compose fine on HTP —
>    it is the *total* single-context size that exceeds the HTP
>    finalization budget. The fix is to split the model and compile
>    weight-shared EPContext binaries (`StaticLLM` / `SplitModel` +
>    `EPContextBinaryGenerator(weight_sharing=True)`).
> 2. **`StaticLLM` expects the ORT-GenAI static-KV contract.** Olive's
>    QNN `StaticLLM` path assumes a GQA model with `past_seq_len` /
>    `total_seq_len` scalar inputs and a fixed sliding-window KV buffer.
>    mobius's `--ep qnn` output uses standard decomposed attention with
>    `attention_mask` / `position_ids` and dynamic concat-grow KV, so it
>    does not match. Implementing gemma4 `--static-cache` (see
>    [Limitations](#limitations)) is the prerequisite that closes this gap.

## Approach (HF-source path)

All four Gemma 4 components (decoder, embedding, vision_encoder,
audio_encoder) are compiled into QNN EPContext binaries together.
Olive's per-component dispatch on `CompositeModelHandler` runs each
pass on every component, then `EPContextBinaryGenerator` and
`ComposeOnnxModels` (both `_accepts_composite_model = True`) finalize
the multimodal package.

## Pipeline

```
HfModel (multimodal Gemma 4)
   ↓ MobiusBuilder (fp32)               4 ONNX components + genai_config + tokenizer + processors
   ↓ OnnxKQuantQuantization (INT4)      mobius-standard Q4_K_M; weights → com.microsoft::MatMulNBits
   ↓ MatMulNBitsToQDQ                   MatMulNBits → MatMul + DequantizeLinear (QNN-compatible QDQ)
   ↓ OnnxStaticQuantization             activations uint16 / weights uint8 (calibrated)
   ↓ StaticLLM                          static shapes for QNN
   ↓ EPContextBinaryGenerator           HTP EPContext blobs (per component, weight-shared)
   ↓ ComposeOnnxModels                  final package
```

Why both `OnnxKQuantQuantization` and `MatMulNBitsToQDQ`?
`OnnxKQuantQuantization` emits `com.microsoft::MatMulNBits`, which has
fast CPU / CUDA kernels but is *not* in the QNN EP's supported-op list
— without `MatMulNBitsToQDQ` the QNN partitioner rejects every
quantized MatMul and the model silently falls back to CPU.
`MatMulNBitsToQDQ` rewrites each `MatMulNBits` into a standard
`MatMul + DequantizeLinear` pair so QNN can claim and compile the
subgraph onto HTP.

## Prerequisites

### Quantization environment (x64, GPU recommended)
```bash
pip install -r requirements.txt
pip install cupy-cuda12x   # accelerates OnnxKQuantQuantization (19–51× speedup)
```

### AOT compilation environment (separate venv, x64 with QNN SDK)
```bash
pip install olive-ai mobius-ai
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
```

Replace `/path/to/qnn/env/bin` in `config.json` with the directory
containing your QNN venv's Python executable.

### Inference (Snapdragon device)
```bash
pip install onnxruntime-qnn onnxruntime-genai
```

## Build

Run `olive run` from the **quantization environment** (not the QNN AOT
venv). Olive invokes the QNN AOT venv automatically via the
`python_environment_path` configured under `systems.qnn_system` for the
`EPContextBinaryGenerator` pass:

```bash
olive run --config config.json
```

## Why no `GraphSurgeries` here?

Most existing QNN recipes (Phi-3, Qwen) chain surgeries like
`RemoveRopeMultiCache`, `AttentionMaskToSequenceLengths`,
`SimplifiedLayerNormToL2Norm`. Those rewrite ModelBuilder-specific
sub-graphs into shapes HTP can lower:

| Surgery | What it does | Why ModelBuilder needs it |
|---|---|---|
| `SimplifiedLayerNormToL2Norm` | `com.microsoft.SimplifiedLayerNorm` → `LpNormalization * gamma` | HTP has no SimplifiedLayerNorm kernel |
| `RemoveRopeMultiCache` | Drop one of ModelBuilder's two RoPE caches | HTP can't dispatch on cache selector |
| `AttentionMaskToSequenceLengths` | `GQA(attention_mask=[B,T])` → `GQA(past_seq_len, total_seq_len)` | HTP's GQA kernel wants scalar seq lens |

`MobiusBuilder` emits opset-23 standard ops (`RMSNormalization`,
`Attention`) instead of the contrib variants, so these surgeries are
either no-ops or inapplicable. Gemma-4–specific surgeries may still be
needed (e.g. lowering the final logit soft-cap `cap * tanh(x / cap)`),
but the existing borrowed-from-Phi-3 set is not it.

## Limitations

This recipe has **not yet been validated end-to-end**. Concrete findings
from HTP experiments on a Snapdragon X (ORT 1.27, onnxruntime-qnn 2.4.0,
QNN SDK 2.48.40, HTP V73):

0. **gemma4 `--static-cache` — DONE** (onnxruntime/mobius, commit `5d804b7`).
   `Gemma4TextAttention`/`Gemma4DecoderLayer` now implement the
   `StaticCacheState` dispatch: static-cache attention bias for both sliding
   and full layer types, per-cache-layer buffers (KV-shared layers borrow the
   source layer's static buffer, so only the `num_kv_layers` cache-owning
   layers get buffers), and dual head_dim (sliding vs full). Verified: static
   decode logits match the dynamic reference at every position (~1e-6 fp32),
   and the full E2B static model **generates correct text on ORT CPU**
   (`build-gguf ... --static-cache --max-seq-len 512` → "The capital of France
   is **Paris**."). `build-gguf --static-cache` emits a fully static-shaped,
   decomposed, QDQ graph (0 fused `Attention`, 0 `MatMulNBits`, 30
   `TensorScatter`, 15 cache layers with dual head_dim 256/512).

1. **Full-model single-graph HTP still needs splitting — two confirmed
   blockers** (Snapdragon X, ORT 1.27, onnxruntime-qnn 2.4.0):
   - **4.7 GB per-layer embedding overflows QNN's 32-bit static-tensor
     size** (`embed_tokens_per_layer` `[262144, 8960]` fp16 = 4 697 620 480 B
     wraps mod 2³² → 402 653 184 B → `Data length mismatch for static
     tensor`). Fix: keep the embedding tables on CPU. Splitting the graph
     after the two embedding `Gather`s yields a 5.5 GB embedding model (CPU)
     and a 2.15 GB transformer (no > 4 GB tensors) — the transformer then
     *composes* on HTP.
   - **The 35-layer transformer fails single-graph HTP finalization**
     (`Failed to finalize QNN graph` after ~20 min) — one context is too
     large. A **4-layer chunk composes on HTP in ~21 s and runs**, so the fix
     is to split the transformer into weight-shared EPContext chunks
     (`SplitModel` / `EPContextBinaryGenerator(weight_sharing=True)`).

2. **Performance: mobius's decompose+inline QDQ form runs slowly on HTP.**
   The 4-layer chunk decodes at ~650–810 ms/step (extrapolates to ~0.2 tok/s
   for 35 layers) because the decomposed SDPA + `DequantizeLinear`+`MatMul`
   land largely on the CPU EP (QNN partitions them out) with HTP↔CPU
   transfers, and the runtime INT4 nibble-unpack (`BitwiseAnd`/`BitShift`,
   constant-foldable) adds overhead. A *reasonable* NPU throughput needs the
   Olive static-INT path: **keep `MatMulNBits`** in the mobius export and run
   `MatMulNBitsToQDQ` (clean int4 initializers, no runtime unpack) +
   `OnnxStaticQuantization` (uint16 activations / uint8 weights) so the matmuls
   hit HTP's integer engine, then chunk into weight-shared EPContext binaries.
   This is now unblocked by the static-cache support above.

3. **HTP op coverage — DONE in the mobius qnn lowering** (onnxruntime/mobius
   commit `be98116`). Empirically (Snapdragon X, onnxruntime-qnn 2.4.0,
   `session.disable_cpu_ep_fallback=1`) the QNN HTP backend has **no kernel**
   for these ops, each forcing its node onto CPU: the fused opset-24
   `Attention`, `com.microsoft::MatMulNBits`, `RotaryEmbedding`,
   `TensorScatter`, `Tile`, `Range`, `BitwiseAnd`/`BitShift`. mobius `--ep qnn`
   now lowers **all** of them to HTP-supported equivalents:
   `Attention`→SDPA, `RotaryEmbedding`→rotate-half, `TensorScatter`→`ScatterND`,
   `Tile`→`Expand`, `Range`→`Constant`+`Slice`, `MatMulNBits`→QDQ, and the
   `BitwiseAnd`/`BitShift` nibble-unpack constant-folds away. HTP-OK primitives
   confirmed: `MatMul`, `Softmax`, `RMSNormalization`, `Gelu`, `Sigmoid`,
   `Transpose`, `Reshape`, `Slice`, `Concat`, `Expand`, `ScatterND`, `Where`,
   `GreaterOrEqual`, `Less`, `And`, `Neg`, `Mul`, `Add`, `Sub`, `Div`,
   `ReduceMean`, `Sqrt`, `Gather`. **Result: a non-quantized (fp16) gemma4
   static model composes and runs *entirely* on the HTP** (no CPU fallback).
   Filed the original fused-op gap as
   [onnxruntime/onnxruntime-qnn#646](https://github.com/onnxruntime/onnxruntime-qnn/issues/646).

4. **INT4 weights still need quantized *activations* for HTP.** A weight-only
   `DequantizeLinear`→`MatMul` (float activations) runs on HTP **only** for
   *per-tensor* weights; *per-channel* and *blocked* int8/int4 weight DQ is
   CPU-forced. QNN HTP claims per-channel/blocked weight matmuls only inside a
   **full int QDQ group** — `DequantizeLinear(act)` + `DequantizeLinear(weight
   per-channel)` → `MatMul` → `QuantizeLinear`. So the GGUF INT4 weights need
   **quantized activations** (`OnnxStaticQuantization`, which needs calibration)
   to keep their matmuls on HTP — that is an Olive pass, not a mobius lowering.
   Two viable deployments: (a) fp16 weights → fully on HTP today via the mobius
   qnn lowering (4× larger, so still needs the transformer chunked for HTP
   finalization); (b) INT4 + Olive static activation quant → int8 QDQ on HTP.

5. **Logit soft-cap may not lower to HTP.** Gemma 4's
   `logits = cap * tanh(logits / cap)` is unusual for QNN. If HTP
   rejects it, options are (a) skip soft-cap during QNN compile and
   apply it in host post-processing, or (b) add a
   `RemoveLogitSoftcap` GraphSurgery upstream in Olive.

6. **Hybrid local/global attention with dual head_dim.** Gemma 4 E2B
   alternates local sliding-window and global attention layers with
   different head_dim. Small dual-head_dim subsets compose on HTP; the
   full alternation at 35 layers is untested past the size blocker above.

7. **256k tokenizer calibration.** `wikitext-2` calibration likely
   under-represents Gemma 4's image / audio special tokens. Consider
   augmenting with multimodal-formatted prompts before production.

8. **`StaticLLM context_length=64`.** Placeholder mirroring existing QNN
   recipes; tune to the target Snapdragon SKU memory budget.

### Reference: tok/s baselines (gemma-4-E2B Q4_K_M, Snapdragon X, 12 threads)

| Runtime | Decode tok/s |
|---|---|
| llama.cpp (CPU ARM64) | 43.7 |
| mobius INT4 ONNX (ORT CPU EP) | 18.2 |
| HuggingFace transformers (fp32 torch CPU) | 0.04 |

## Measured HTP results (fully-on-HTP path)

With the mobius qnn lowering (commit `be98116`) **every op runs on HTP** — no
CPU fallback — once the INT4 GGUF weights are re-quantized to a QNN-runnable
form. Measured on a Snapdragon X (onnxruntime-qnn 2.4.0), 10-layer model at
E2B per-layer dims (hidden 1536, intermediate 6144, head_dim 256/512,
static cache max_seq_len=512), extrapolated to the 35-layer transformer
(embeddings + lm_head excluded, run on CPU):

| Quantization | **HTP (NPU)** decode | **CPU** decode |
|---|---|---|
| int4 per-channel + uint8 activations | **16.4 tok/s** | 17.5 tok/s |
| int8 per-channel + uint8 activations | 16.6 tok/s | 16.7 tok/s |

**The NPU decode speed ≈ CPU (~16–17 tok/s); it does not beat CPU here, and
int4 vs int8 barely differ.** Single-token (batch=1) decode is memory-bound and
low-parallelism — not the regime an NPU wins. IO-binding (in-place KV) and
EPContext were tried and did **not** change decode latency (CPU/HTP share SoC
DRAM, so there is no host↔device transfer to eliminate; EPContext only speeds up
model *loading*). The HTP's real value is **power efficiency, prefill throughput,
and batching**, not single-stream decode tok/s.

**How to get a fully-on-HTP model (fake-calibration shortcut for testing).**
QNN HTP claims a weight `DequantizeLinear`→`MatMul` only for **per-channel**
int4/int8 weights inside a **full int QDQ group** (quantized activations); the
GGUF Q4_K **blocked** form (`block_size=32`) is CPU-forced at any bit width
(see [onnxruntime/onnxruntime-qnn#650](https://github.com/onnxruntime/onnxruntime-qnn/issues/650)).
So the INT4 blocked weights must be re-quantized to **per-channel** and the
activations quantized. For a throughput/latency test where accuracy does not
matter, run `onnxruntime.quantization.quantize_static` with a **fake
calibration reader** (a couple of random/zero input batches) instead of the
expensive wikitext calibration — it inserts per-channel weight QDQ + uint8
activation QDQ and completes in seconds:

```python
from onnxruntime.quantization import (
    quantize_static, QuantType, QuantFormat, CalibrationDataReader)

class FakeCalib(CalibrationDataReader):
    def __init__(self, feeds): self.it = iter(feeds)          # a few dummy input dicts
    def get_next(self): return next(self.it, None)

quantize_static(
    "model_static.onnx", "model_int8.onnx", FakeCalib(dummy_feeds),
    quant_format=QuantFormat.QDQ, per_channel=True,
    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt4,  # or QInt8
    op_types_to_quantize=["MatMul"])
```

The resulting model loads on the QNN HTP with
`session.disable_cpu_ep_fallback = "1"` and runs entirely on the NPU. For a
real deployment, replace `FakeCalib` with a proper calibration dataset (the
`static_quant` pass in `config_gguf.json`) so the output is accurate.

## Discussion

If you have a Snapdragon test rig and the pipeline blows up on a
specific pass, please drop the trace in a comment — this recipe is
intentionally a template for iteration, not a finished product.
