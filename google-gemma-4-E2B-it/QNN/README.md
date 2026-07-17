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

0. **gemma4 `--static-cache` is the top prerequisite.** HTP needs fully
   static shapes, and Olive's QNN `StaticLLM` needs the ORT-GenAI
   static-KV contract. mobius `build-gguf --static-cache` now provides the
   general plumbing, but the Gemma-4 decoder itself does not implement the
   `StaticCacheState` dispatch — its `Gemma4TextAttention` has KV-shared
   layers (some layers borrow K,V from a source layer and keep no cache of
   their own), sliding-window + full alternating layers, and dual
   local/global RoPE. Adding static-cache support means: (a) generating a
   static-cache attention bias for *both* layer types via
   `create_static_cache_attention_bias`, (b) threading `StaticCacheState`
   through `Gemma4DecoderLayer` / `Gemma4TextAttention`, allocating buffers
   only for the `num_kv_layers` cache layers, (c) making KV-shared layers
   read the source layer's static buffer. This is correctness-critical
   surgery on the most complex model in mobius and must be validated with
   HF numerical parity before use.

1. **Full-model single-graph HTP compose fails on size.** See the blocker
   note under [step 2](#gguf-direct-step-2--compile-to-a-qnn-htp-context-binary).
   Needs model splitting + weight-shared EPContext binaries.

2. **Fused ops have no HTP kernel** (filed as
   [onnxruntime/onnxruntime-qnn#646](https://github.com/onnxruntime/onnxruntime-qnn/issues/646)):
   the opset-24 `Attention` op and `com.microsoft::MatMulNBits` are forced
   onto CPU by the QNN EP. mobius `--ep qnn` decomposes / inlines both so
   the graph stays on HTP; empirically `RMSNormalization`, `Gelu`,
   `MatMul`, `Softmax`, `Transpose` and blocked `DequantizeLinear` *do*
   run on HTP.

3. **Logit soft-cap may not lower to HTP.** Gemma 4's
   `logits = cap * tanh(logits / cap)` is unusual for QNN. If HTP
   rejects it, options are (a) skip soft-cap during QNN compile and
   apply it in host post-processing, or (b) add a
   `RemoveLogitSoftcap` GraphSurgery upstream in Olive.

4. **Hybrid local/global attention with dual head_dim.** Gemma 4 E2B
   alternates local sliding-window and global attention layers with
   different head_dim. Small dual-head_dim subsets compose on HTP; the
   full alternation at 35 layers is untested past the size blocker above.

5. **256k tokenizer calibration.** `wikitext-2` calibration likely
   under-represents Gemma 4's image / audio special tokens. Consider
   augmenting with multimodal-formatted prompts before production.

6. **`StaticLLM context_length=64`.** Placeholder mirroring existing QNN
   recipes; tune to the target Snapdragon SKU memory budget.

### Reference: tok/s baselines (gemma-4-E2B Q4_K_M, Snapdragon X, 12 threads)

| Runtime | Decode tok/s |
|---|---|
| llama.cpp (CPU ARM64) | 43.7 |
| mobius INT4 ONNX (ORT CPU EP) | 18.2 |
| HuggingFace transformers (fp32 torch CPU) | 0.04 |

These are the CPU baselines to beat once the HTP/NPU path is unblocked.

## Discussion

If you have a Snapdragon test rig and the pipeline blows up on a
specific pass, please drop the trace in a comment — this recipe is
intentionally a template for iteration, not a finished product.
