# Gemma 4 E2B (google/gemma-4-E2B-it)

Olive recipes that export [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
to ONNX via the [`MobiusBuilder`](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
pass and (optionally) quantize the decoder with Olive's K-Quant (Q4_K_M)
pass for INT4 deployment.

Gemma 4 is an any-to-any multimodal model with vision, audio, and text
capabilities. The pipeline produces four ONNX components (decoder,
vision_encoder, audio_encoder, embedding) for use with ORT GenAI.
`MobiusBuilder` writes a fully-formed ORT GenAI package
(`genai_config.json`, `tokenizer.json`, `image_processor.json`,
`audio_feature_extraction.json`) alongside the ONNX files — no
post-processing required.

## Prerequisites

```bash
pip install olive-ai mobius-ai
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| CPU | `pip install onnxruntime-genai` |
| GPU (CUDA) | `pip install onnxruntime-genai-cuda` |
| OpenVINO (NPU) | `pip install onnxruntime-genai` + [OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) |

## Recipes

| Recipe | Pipeline | Output dir |
|---|---|---|
| `cpu/fp32/config.json` | `MobiusBuilder(fp32)` | `cpu/fp32/models` |
| `cpu/int4/config.json` | `MobiusBuilder(fp32)` → `OnnxKQuantQuantization(bits=4, block=32)` | `cpu/int4/models` |
| `cuda/fp16/config.json` | `MobiusBuilder(fp16)` | `cuda/fp16/models` |
| `cuda/int4/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block=32)` | `cuda/int4/models` |
| `openvino/npu/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block=32)` → `MatMulNBitsToQDQ` | `openvino/npu/models` |

K-Quant (Q4_K_M) is significantly faster with GPU acceleration —
install `cupy-cuda12x` for a 19–51× speedup during quantization.

The `openvino/npu` recipe builds the portable INT4 ONNX and writes a
`genai_config.json` whose `provider_options` select the OpenVINO EP. The
graph is device-independent, so mobius emits a default OpenVINO
`device_type` of `NPU`; to target a different OpenVINO device (`GPU` / `CPU`)
just edit `device_type` in the generated `genai_config.json` — no rebuild is
needed. This requires the OpenVINO EP support in `MobiusBuilder`
([microsoft/Olive](https://github.com/microsoft/Olive)) and the mobius
`openvino` EP ([onnxruntime/mobius](https://github.com/onnxruntime/mobius)).

> **⚠️ Known limitation — does not fully convert on current OpenVINO yet.**
> The recipe removes two classes of OpenVINO-unconvertible ops on our side:
> the mobius `openvino` EP disables `SkipSimplifiedLayerNormalization` fusion
> (keeping `Add` + `RMSNorm` separate), and `MatMulNBitsToQDQ` lowers the INT4
> `MatMulNBits` weights to standard QDQ. Two blockers remain **on the OpenVINO
> side** (frontend fixes pending; tracked upstream):
> 1. `ai.onnx::RMSNormalization` (opset 24) is not implemented by the OpenVINO
>    ONNX frontend.
> 2. `ai.onnx::Attention` (opset 24) with a KV cache fails shape inference in
>    the OpenVINO frontend (past `f16[?,1,?,256]` vs current `f16[?,?,256]`
>    rank mismatch in the KV `Concat`).
>
> Until those land in OpenVINO, `openvino.convert_model` / the OpenVINO EP
> cannot consume the graph. A prebuilt INT4 package for version/hardware
> testing is on the Hub:
> [`justinchuby/gemma-4-E2B-it-ONNX`](https://huggingface.co/justinchuby/gemma-4-E2B-it-ONNX).

## Build

```bash
# CPU, full precision
olive run --config cpu/fp32/config.json

# CPU, INT4 (K-Quant)
olive run --config cpu/int4/config.json

# CUDA, FP16
olive run --config cuda/fp16/config.json

# CUDA, INT4 (K-Quant)
olive run --config cuda/int4/config.json

# OpenVINO NPU, INT4 (K-Quant)
olive run --config openvino/npu/config.json
```

Each command produces the full ORT GenAI package in the recipe's
`output_dir`:

```
<output_dir>/
├── decoder/model.onnx          # Text decoder
├── vision_encoder/model.onnx   # Vision encoder
├── audio_encoder/model.onnx    # Audio encoder
├── embedding/model.onnx        # Embedding fusion
├── genai_config.json           # Runtime configuration
├── image_processor.json
├── audio_feature_extraction.json
├── tokenizer.json
└── tokenizer_config.json
```

## Inference

```bash
# Text-only (CPU, fp32)
python inference.py --prompt "What is the capital of France?"

# CPU INT4
python inference.py --variant int4 --prompt "Hello"

# CUDA INT4
python inference.py --device gpu --variant int4 --prompt "Explain quantum computing"

# Interactive mode
python inference.py --device gpu --variant int4 --interactive
```

## Evaluation

```bash
# MMLU Pro (default 100 samples), CPU
python eval.py

# CUDA INT4
python eval.py --device gpu --variant int4
```

## QNN (Snapdragon NPU) — experimental, two-step

> **Status: untested on hardware.** These configs are authored from the Olive
> QNN/HTP reference flow. Building an HTP context binary requires the **QNN SDK
> (via `onnxruntime-qnn`) and a Snapdragon target** — it cannot run on a
> generic x86/CUDA host. Validated on CPU: the mobius build, K-Quant, the
> `MatMulNBitsToQDQ` INT4-QDQ lowering (produces zero `com.microsoft` ops), and
> the decoder `GraphSurgeries`. The HTP-specific steps
> (`OnnxStaticQuantization` for HTP, `SplitModel`, `StaticLLM`,
> `EPContextBinaryGenerator`) are **not yet verified on hardware**.

Because the QNN/HTP LLM passes (`SplitModel`, `StaticLLM`,
`EPContextBinaryGenerator`) operate on a single decoder — and mobius emits a
multi-component package — the QNN path is a **two-step** flow that compiles
each component separately:

```bash
# Step 1 — build the portable (onnx-standard) multi-component model
olive run --config qnn/build/config.json

# Step 2 — compile each component to an HTP context binary
#   (run in a Python env that has onnxruntime-qnn installed; set
#    python_environment_path and soc_model for your Snapdragon target)
olive run --config qnn/decoder/config.json          # text decoder (full HTP LLM flow)
olive run --config qnn/vision_encoder/config.json   # vision encoder
olive run --config qnn/audio_encoder/config.json    # audio encoder
olive run --config qnn/embedding/config.json        # embedding fusion
```

| Recipe | Component | Pipeline |
|---|---|---|
| `qnn/build/config.json` | all | `MobiusBuilder(fp32, onnx-standard)` → `OnnxKQuantQuantization(int4)` → decoder + vision + audio + embedding |
| `qnn/decoder/config.json` | decoder | `MatMulNBitsToQDQ` → `GraphSurgeries` → `OnnxStaticQuantization` → `SplitModel` → `StaticLLM` → `EPContextBinaryGenerator` → `ComposeOnnxModels` |
| `qnn/{vision_encoder,audio_encoder,embedding}/config.json` | encoders | `MatMulNBitsToQDQ` → `EPContextBinaryGenerator(enable_htp_fp16_precision)` → `ComposeOnnxModels` |

Notes / TODOs before hardware bring-up:
- Set `systems.qnn_system.python_environment_path` to a venv with
  `onnxruntime-qnn` installed, and set `cb.provider_options.soc_model` to your
  device's SoC id.
- The build step quantizes weights to INT4 with K-Quant; `MatMulNBitsToQDQ`
  then lowers the `MatMulNBits` ops to **standard-ONNX INT4 QDQ**
  (`DequantizeLinear` + `MatMul`, zero `com.microsoft` ops), which the QNN HTP
  backend can consume. The decoder additionally static-quantizes activations
  (`wikitext` calibration); the encoders keep FP16 activations on HTP.
- Adjust `context_length` / `batch_size` in `StaticLLM` for your latency budget.
- Only `AttentionMaskToSequenceLengths` and `SimplifiedLayerNormToL2Norm`
  surgeries apply to the mobius decoder — the ModelBuilder-specific
  `RemoveRopeMultiCache` surgery is intentionally omitted (it does not match
  mobius's single-`RotaryEmbedding` structure).

## References

- Mobius docs: <https://github.com/onnxruntime/mobius>
- Olive `MobiusBuilder` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py>
- Olive `OnnxKQuantQuantization` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/kquant_quantization.py>
