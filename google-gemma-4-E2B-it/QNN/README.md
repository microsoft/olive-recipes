# Gemma 4 E2B QNN (Snapdragon Hexagon NPU) recipe

> **Status: WORK IN PROGRESS / EXPLORATORY.** This recipe is a starting
> point for running [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
> on the Qualcomm Hexagon NPU via the QNN ONNX Runtime execution
> provider (Snapdragon X / Copilot+ PC, Snapdragon 8 Gen 3+, etc.).
> It has *not* yet been end-to-end validated on hardware — see the
> [Limitations](#limitations) section below before using it in
> production.

This recipe targets the **text decoder only**. Gemma 4's vision and
audio encoders run on CPU (`google-gemma-4-E2B-it/cpu/`) or GPU
(`google-gemma-4-E2B-it/cuda/`); only the LM decoder is compiled into
an EPContext binary for HTP execution.

## Pipeline overview

```
HfModel (multimodal Gemma4)
   ↓ MobiusBuilder fp32       → ORT GenAI multi-component package
   ↓ OnnxKQuantQuantization   → INT4 weights (decoder only)
   ↓ MatMulNBitsToQDQ         → QDQ format for static quantization
   ↓ GraphSurgeries           → QNN-friendly graph (Rope unmerge, mask, L2Norm)
   ↓ OnnxStaticQuantization   → activations uint16 / weights uint8 (calibrated)
   ↓ SplitModel + StaticLLM   → static-shape sub-graphs for QNN
   ↓ EPContextBinaryGenerator → compiled HTP EPContext blob
```

## Prerequisites

### Quantization environment (x64 with CUDA GPU)
Quantization (especially `OnnxStaticQuantization`) is resource-intensive
and accelerated by GPU:

```bash
pip install -r requirements.txt
```

### AOT compilation environment (separate venv, x64 with QNN SDK)
Compilation into the EPContext binary requires
`onnxruntime-qnn`:

```bash
pip install olive-ai mobius-ai
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple "onnxruntime-qnn" --no-deps
```

Set `/path/to/qnn/env/bin` in `config.json` to the directory containing
your QNN venv's Python executable.

### Inference environment (Snapdragon device)
On Copilot+ PC / Snapdragon X / Android (Snapdragon 8 Gen 3+):

```bash
pip install onnxruntime-qnn onnxruntime-genai
```

## Build

```bash
olive run --config config.json
```

Output is a self-contained EPContext binary package suitable for QNN HTP
execution.

## Limitations

This recipe has **not yet been validated end-to-end**. Known gaps:

1. **Multimodal `google/gemma-4-E2B-it` always produces a 4-component
   package** (decoder + embedding + vision + audio). MobiusBuilder
   currently does not expose a `model_type` / `module_class` override
   to force the text-only `gemma4_text` build, so the pipeline must
   either: (a) run the QNN passes only on the `decoder` component and
   leave the others as fp32, or (b) wait for the upstream MobiusBuilder
   to gain a `module_class` parameter. This recipe assumes (a) but the
   QNN pass chain currently expects a single ONNX model — the
   integration is still TODO.

2. **Gemma 4's exotic ops may break QNN GraphSurgeries.** The recipe
   borrows surgeries (`RemoveRopeMultiCache`, `AttentionMaskToSequenceLengths`,
   `SimplifiedLayerNormToL2Norm`) from Phi-3 / Qwen QNN recipes; they
   have not been verified against Gemma 4's hybrid local/global
   attention, `tie_word_embeddings`, dual head_dim KV cache, or final
   logit soft-capping (`logits = cap * tanh(logits / cap)`). The
   soft-cap subgraph in particular may not lower cleanly to HTP — may
   need to be folded into the logit lookup or skipped during QNN
   compilation.

3. **Per-layer-input embeddings.** Gemma 4 E2B emits a second embedding
   output (`per_layer_inputs`, shape `[B, S, L*D]`) consumed by every
   decoder block. The split between embedding-on-CPU and decoder-on-HTP
   needs a custom orchestrator (or both sub-models compiled into QNN
   together).

4. **Calibration data shape.** `OnnxStaticQuantization` calibration uses
   `wikitext-2`; for Gemma 4 (which has a 256k tokenizer including
   image / audio special tokens) the calibration set may under-represent
   tokens that actually appear at inference time. Consider augmenting
   with multimodal-formatted prompts.

5. **`StaticLLM context_length=64`.** Mirrors Phi-3 / Qwen QNN recipes
   but is unlikely to be useful for a real Gemma 4 deployment; tune to
   target Snapdragon SKU memory budget once HW validation is possible.

## Discussion

If you have a Snapdragon test rig and run into specific failures,
please add a comment with the trace; this recipe is intended as a
template that other contributors can iterate on rather than a
production-ready pipeline.
