# Qwen3.5-0.8B: Mobius export, then multi-component Vitis quantization

This recipe follows the **export-first VLM flow** in
[PR #524](https://github.com/microsoft/olive-recipes/pull/524).
Export the complete `Qwen/Qwen3.5-0.8B` VLM with Mobius, then use **one Olive
configuration with independent `builds.components` pipelines** to quantize the
ONNX decoder and vision encoder. There is no HF-stage quantization, ModelBuilder,
or ortpackage generation.

| Component | Weight quantization | Activation quantization | Calibration |
|-----------|---------------------|-------------------------|-------------|
| `decoder` | UINT8 static quantization | UINT16 | WikiText-2 train, 128 samples, up to 512 tokens each |
| `vision_encoder` | Symmetric INT4 RTN, block 128, converted to QDQ | None; FP32 | None |
| `embedding` | Unchanged FP32 | None | None |

The output is a **Vitis-oriented quantized ONNX model directory**, not a compiled
NPU binary. CPU is used for export/calibration, not as a claim that every final
INT4/UINT16 QDQ operator has a CPU kernel. Running on an AMD NPU additionally
requires a compatible Ryzen AI/Vitis AI runtime and support for Qwen3.5's
GatedDeltaNet operators and recurrent states. Vitis metadata does not establish
that hardware support.

## 1. Install dependencies

Use Python 3.10 or newer in an isolated environment:

```powershell
pip install -r requirements.txt
```

The source revisions include Olive's multi-component builds and Mobius's
Qwen3.5 support. Use the ONNX Runtime build required by your Ryzen AI installation
for deployment; do not replace its provider-enabled runtime with the CPU wheel.

Run all commands below from `Qwen-Qwen3.5-0.8B\vitis`.

## 2. Export the complete FP32 VLM

```powershell
olive capture-onnx-graph --model_name_or_path Qwen/Qwen3.5-0.8B --use_mobius_builder --precision fp32 --output_path exported_model
```

Keep all three components. The decoder consumes embeddings from the exported
embedding model, including during text calibration:

```text
exported_model/
  decoder/model.onnx
  vision_encoder/model.onnx
  embedding/model.onnx
  genai_config.json
  tokenizer and processor files
```

Do not use a text-only export, a pre-quantized export, or an FP16 export with this
configuration. ONNX external-data files must remain alongside their graphs.

## 3. Quantize the ONNX components

```powershell
python optimize.py
```

`optimize.py` calls `olive.run` once with `config.json`. The configuration contains:

```json
{
    "builds": {
        "_default": {"output_dir": "optimized_model"},
        "decoder": {
            "components": ["decoder"],
            "pipeline": ["gs", "sq", "text_metadata"]
        },
        "vision_encoder": {
            "components": ["vision_encoder"],
            "pipeline": ["gemm_to_matmul", "vision_rtn", "mq", "vision_metadata"]
        }
    }
}
```

There is no third quantization build for `embedding`: the helper retains that
component unchanged and copies the exported tokenizer, processor and
`genai_config.json`. The pipelines preserve component input/output names and
floating-point interfaces. Use a fresh output directory on subsequent runs;
the helper refuses to overwrite an existing output.

To run only the Olive builds without assembling the unchanged runtime assets:

```powershell
olive run --config config.json
```

### Decoder calibration

`wikitext2_train` uses `Salesforce/wikitext`, subset `wikitext-2-raw-v1`, split
`train`, with line-by-line tokenization and no added special tokens.

A normal text dataloader is insufficient for this exported VLM: its decoder
takes `inputs_embeds`, three-axis MRoPE positions, full-attention KV caches,
and GatedDeltaNet convolution/recurrent states, not just `input_ids`.
`user_script.py` adapts Olive's tokenized WikiText batches to that contract:

1. Run each text token through the original FP32 embedding component, with empty image features.
2. Supply text-only MRoPE positions and the hybrid attention states.
3. Advance the original FP32 decoder to obtain the next token's real states.

Samples start with empty KV caches and zero recurrent states. Tokens are
processed sequentially, so a 512-token sample can require 512 calibration steps;
the full 128-sample run is substantially more expensive than ordinary batched
text calibration. Lower `max_samples` and `max_seq_len` for a first trial, then
restore them for the full run. This is **text-only** activation calibration, not
image-conditioned decoder calibration.

The fused normalization operators left after graph surgery are excluded from
static quantization: their optional statistics outputs are not produced by the
CPU kernel and cannot be reduced during calibration. `quant_preprocess` is
disabled because ORT's symbolic preprocessor does not handle this Mobius hybrid
graph; it is not necessary to repeat export-time graph processing.

### Vision quantization

The vision build converts constant-weight Gemm operations to MatMul/Add, applies
INT4 blockwise RTN, converts MatMulNBits weights to QDQ, and adds Vitis metadata.
It has no WikiText/image dataset or static activation pass. Vision activations
remain FP32, so its metadata intentionally does not advertise UINT16 activations.
The small RTN intermediate is saved inline so the pinned QDQ conversion pass
retains the vision tower's unquantized tensors rather than stale external-data
offsets. `mq` saves the final vision graph with external data again.

## Changes from the single-text-model reference

| Reference pass | Treatment in this export-first recipe |
|----------------|---------------------------------------|
| `QuaRot` | Removed: it rotates PyTorch/HF weights before export, not an already-exported ONNX component. Metadata therefore does not claim QuaRot. |
| `CaptureSplitInfo` | Removed: this is an HF/PyTorch split-metadata pass. The ONNX components already exist. |
| `ModelBuilder` | Removed: Mobius performs the export in step 2. The decoder is quantized once by the WikiText-calibrated static quantizer, not additionally quantized to INT4. |
| `MatMulNBitsToQDQ` | Retained for the RTN vision component, with INT4 and explicit zero points. The decoder static quantizer directly produces QDQ. |
| `GraphSurgeries` | Retains `SimplifiedLayerNormToL2Norm` on the decoder; adds Gemm conversion on vision. |
| `RemoveRopeMultiCache` | Removed: this targets ModelBuilder's specific multi-cache/If layout, not Mobius's MRoPE graph. |
| `AttentionMaskToSequenceLengths` | Removed: changing only GQA inputs is not appropriate for the hybrid decoder; preserve its attention-mask input and exported runtime mappings. |
| `OnnxStaticQuantization` | Retained on the decoder only: WikiText, UINT16 activations, UINT8 for remaining eligible weights, CPU calibration, and the reference operator exclusions. |
| `VitisAIAddMetaData` | Retained with separate, accurate text/vision quantization metadata. |
| `SplitModel`, `StaticLLM` | Removed: the reference splits a language model into embedding/transformer/lm-head stages. Those are not the VLM's decoder/vision/embedding components, and StaticLLM's GQA assumptions do not cover hybrid recurrent state. |

No hard-coded `/lm_head/MatMul_Q4` exclusion is carried over: that is a
ModelBuilder node name, not a guaranteed Mobius node name, and the decoder does
not use INT4 in this recipe. Static quantization covers its eligible operators;
ineligible operators and the unchanged embedding stay floating point.

## Output and deployment boundary

```text
optimized_model/
  decoder/model.onnx          # WikiText-calibrated decoder
  vision_encoder/model.onnx   # RTN vision encoder
  embedding/model.onnx        # unchanged FP32 embedding
  genai_config.json
  tokenizer and processor files
```

External tensor data accompanies each graph. Keep the directory together.
`cache` contains reusable Olive pass artifacts; it is not the deployment output.
`max_concurrent_builds` is set to one to avoid running the two memory-intensive
component pipelines simultaneously.

The reference's `StaticLLM` context length of 128 and Vitis session options are
not copied into a mismatched VLM graph. This recipe preserves the dynamic,
stateful Mobius interfaces; it does not generate static prefill/decode graphs,
compile a Vitis context binary, or establish NPU inference support. Hardware
compilation and text/image inference with the target runtime remain required
before treating these artifacts as a deployable Vitis model.
