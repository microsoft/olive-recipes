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

This emits `model_int4/model.onnx` — the Gemma 4 E2B text decoder with
INT4 `MatMulNBits` projections (205 nodes), opset-24 standard `Attention`
(35, no GroupQueryAttention — the QNN capability advertises empty
`gqa_dtypes`), and `RMSNormalization`. Mixed Q6_K tensors in the Q4_K_M
preset are re-quantized to 4-bit / block-32 automatically.

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

`config_gguf.json` drops `MobiusBuilder` **and** `OnnxKQuantQuantization`
(the INT4 weights already exist) and runs only
`MatMulNBitsToQDQ → OnnxStaticQuantization → StaticLLM →
EPContextBinaryGenerator`.

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

This recipe has **not yet been validated end-to-end**. Known gaps:

1. **Logit soft-cap may not lower to HTP.** Gemma 4's
   `logits = cap * tanh(logits / cap)` is unusual for QNN. If HTP
   rejects it, options are (a) skip soft-cap during QNN compile and
   apply it in host post-processing, or (b) add a
   `RemoveLogitSoftcap` GraphSurgery upstream in Olive.

2. **Hybrid local/global attention with dual head_dim.** Gemma 4 E2B
   alternates local sliding-window (head_dim=256) and global
   (head_dim=512) attention layers. Whether HTP can dispatch this
   correctly per-layer needs testing.

3. **`per_layer_inputs` data flow.** The embedding component emits a
   second output (`per_layer_inputs`, shape `[B, S, L*D]`) consumed by
   every decoder block. When all components compile to QNN this should
   "just work" (the data path stays inside the package), but the
   `StaticLLM` pass may need a hint to recognise this extra tensor.

4. **256k tokenizer calibration.** `wikitext-2` calibration likely
   under-represents Gemma 4's image / audio special tokens. Consider
   augmenting with multimodal-formatted prompts before production.

5. **`StaticLLM context_length=64`.** Placeholder mirroring existing QNN
   recipes; tune to target Snapdragon SKU memory budget.

6. **Standard `Attention` op, not `GroupQueryAttention`.** mobius only
   emits `com.microsoft::GroupQueryAttention(seqlens_k, total_seq_len)`
   when the EP capability advertises `gqa_dtypes`. The QNN EP
   capability in mobius currently has an empty `gqa_dtypes` list, so
   `Gemma4TextModel.forward` (`src/mobius/models/gemma4.py:1500-1508`)
   falls back to the standard opset-23 `Attention` with an
   `attention_mask` input. QNN's HTP backend should have an attention
   kernel for the standard op, but if it doesn't lower well there are
   two options:
   - extend mobius `ep_capabilities()` to advertise QNN-supported
     dtypes for `gqa_dtypes`, then mobius will emit `GQA` directly
     (no GraphSurgery needed); or
   - port `AttentionMaskToSequenceLengths` to operate on standard
     `Attention` (it currently checks for `GroupQueryAttention` only
     and no-ops otherwise).

## Discussion

If you have a Snapdragon test rig and the pipeline blows up on a
specific pass, please drop the trace in a comment — this recipe is
intentionally a template for iteration, not a finished product.
