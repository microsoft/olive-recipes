# Gemma 4 E2B (google/gemma-4-E2B-it)

Olive recipes that export [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
to ONNX via the [`MobiusBuilder`](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
pass. The latest `int4-optimized` recipes use symmetric RTN INT4 block 32 for
the decoder, including `lm_head`, and RTN INT8 for embedding, vision, and audio.
They also omit unused shared-KV cache outputs and configure device-specific
runtime settings. Full-precision, K-Quant INT4, and mixed INT4/INT8 recipes
remain available as alternatives.

Gemma 4 is an any-to-any multimodal model with vision, audio, and text
capabilities. The pipeline produces four ONNX components (decoder,
vision_encoder, audio_encoder, embedding) for use with ORT GenAI.
`MobiusBuilder` writes a fully-formed ORT GenAI package
(`genai_config.json`, `tokenizer.json`, `image_processor.json`,
`audio_feature_extraction.json`) alongside the ONNX files — no
post-processing required.

## Prerequisites

```bash
pip install -r requirements.txt
```

For the standard recipes, install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| CPU | `pip install onnxruntime-genai` |
| GPU (CUDA) | `pip install onnxruntime-genai-cuda` |

The optimized recipes require the patched Olive, ORT, and compatible GenAI
builds described below; installing these pip packages alone is not sufficient.

## Recipes

| Recipe | Pipeline | Output dir |
|---|---|---|
| `cpu/int4-optimized/{export,text,embedding,vision,audio}.json` | FP32 export, RTN INT4 decoder + INT8 components, shared-KV output omission | `cpu/int4-optimized/models` |
| `cuda/int4-optimized/{export,text,embedding,vision,audio}.json` | FP16 export, RTN INT4 decoder + INT8 components, shared-KV output omission | `cuda/int4-optimized/models` |
| `cpu/fp32/config.json` | `MobiusBuilder(fp32)` | `cpu/fp32/models` |
| `cpu/int4/config.json` | `MobiusBuilder(fp32)` → `OnnxKQuantQuantization(bits=4, block=32)` | `cpu/int4/models` |
| `cuda/fp16/config.json` | `MobiusBuilder(fp16)` | `cuda/fp16/models` |
| `cuda/int4/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block=32)` | `cuda/int4/models` |

### Optimized INT4 Requirements

The opt-in `cpu/int4-optimized` and `cuda/int4-optimized` recipes use INT4
decoder weights, INT8 embedding/vision/audio weights, and shared-KV output
omission. Standard and mixed recipes remain unchanged.

**Patched builds required:** Olive must provide `MobiusBuilder.genai_config`
and `RemoveUnusedGQACacheOutputs`. ORT must support borrowed past KV buffers
with omitted GQA cache outputs on the selected device; CUDA additionally
requires H512 XQA and fpA-intB GEMM support. Use an ABI-compatible ORT GenAI
build. Stock pip packages alone do not guarantee these features.

For CUDA, build ORT with `onnxruntime_USE_FPA_INTB_GEMM=ON` and
`onnxruntime_USE_FPA_INTB_GEMM_FULL=OFF`. Symmetric RTN INT4 block 32 uses the
existing compact kernels; no asymmetric kernel additions are required. The
optimized decode path also requires the symmetric INT4 block-32 M=1
tile tuning (two columns, 256 threads). Keep `ep.cuda.fpa_intb_gemm=1` enabled.

Run the commands below from `google-gemma-4-E2B-it`. Export first, then
quantize the four components into the same standalone package. Repeat the
full sequence when regenerating; do not requantize an already-quantized model.
No assembly script is needed. Both exports enable shared past/present buffers.

### Optimized CPU INT4

FP32 export with symmetric RTN INT4 block 32 and `accuracy_level=4` for the
decoder; symmetric RTN INT8 block 128 for embedding, vision, and audio.
Activations and KV caches remain FP32 (`cache_type: "float32"`); INT8 activation
compute does not imply INT8 cache storage.

```bash
olive run --config cpu/int4-optimized/export.json
olive run --config cpu/int4-optimized/text.json
olive run --config cpu/int4-optimized/embedding.json
olive run --config cpu/int4-optimized/vision.json
olive run --config cpu/int4-optimized/audio.json
```

All components use 12 intra-op threads and one inter-op thread, tuned on AMD
EPYC 7V12. Adjust for other hardware; NUMA affinity is a separate launch setting:

```bash
numactl --cpunodebind=0 --membind=0 python inference.py \
	--model-path cpu/int4-optimized/models --prompt "Hello"
```

### Optimized CUDA INT4

FP16 export with symmetric RTN INT4 block 32 for the decoder and
asymmetric RTN INT8 block 32 for embedding, vision, and audio. Activations
and KV caches remain FP16 (`cache_type: "float16"`). Runtime settings enable
CUDA graphs and fpA-intB GEMM with profile shapes `1,729`.

**Additional runtime fixes required:** ORT's fpA-intB profiler
must use checked buffer offsets for large vocabulary projections, and ORT
GenAI must preserve static decode embedding buffers across prefills. Without
these fixes, long prompts can crash and graph-enabled continuation can produce
corrupted output. Include the [ORT CUDA RMSNorm alignment guard](https://github.com/microsoft/onnxruntime/pull/33178)
to avoid CUDA error 716 with two-byte-offset FP16 buffers.

Use the same GPU visibility for all commands:

```bash
olive run --config cuda/int4-optimized/export.json
olive run --config cuda/int4-optimized/text.json
olive run --config cuda/int4-optimized/embedding.json
olive run --config cuda/int4-optimized/vision.json
olive run --config cuda/int4-optimized/audio.json

python inference.py --model-path cuda/int4-optimized/models --prompt "Hello"
```

The decoder, including `lm_head`, remains INT4. Validate the exported package
and your target workload when changing exporter or runtime versions. CUDA
validation is limited to A100; multimodal persistent-cache continuation has
not been validated. Performance tuning and quantization do not guarantee
exact-output compliance or quality equivalence to the original model.

### Mixed quantization (separate text / vision / audio / embedding)

These recipes split the model into components with per-component
quantization — a **mixed int4/int8 text decoder**, int8 for the vision and
audio encoders, and int8 for the token embedding — for better accuracy vs.
latency/size trade-offs.

**Mixed-bit decoder**: the decoder is int4 K-Quant by default, but the most
quantization-sensitive weights are upcast to int8 via the
`customized_weight_config` in `text.json`. This targets, in every one of the
35 transformer layers, the `down_proj`, `gate_proj`, `up_proj`, and
`o_proj` MatMuls plus the global `lm_head` (141 weights total → int8; the
remaining `q/k/v_proj` and per-layer gates stay int4). Empirically these
nodes carry most of the int4 accuracy loss, so upcasting only them recovers
most of the fp16 quality for a small size cost (CUDA decoder 1.41 GB pure-int4
→ 2.50 GB mixed).

Validation (CUDA, full eval sets), including the optimized INT4 decoder:

| Metric | int4 decoder | int4-optimized decoder | mixed int4/int8 decoder | PyTorch bf16 |
|---|---|---|---|---|
| AI2D exact_match (3,088) | 57.7% | 60.72% | 62.86% (+5.2) | 63.6% |
| FLEURS en_us strict WER (647) | 9.48% | 9.54% | 8.94% (−0.54) | 10.5% |
| MMLU 5-shot (14,042) | — | 57.87% | 60.09% | 60.8% |
| decoder size | 1.41 GB | 1.48 GB | 2.50 GB | — |

The `int4-optimized` scores are from a separate 2026-10-02 full-set run on
A100 with patched ORT 1.31.0 and GenAI 0.15.2, not a matched rerun of the other
columns. Optimized AI2D uses answer-choice extraction. Optimized decoder size
includes the graph and external weights (decimal GB).

> Note: `customized_weight_config` keys are exact exported node names
> (e.g. `.../down_proj/MatMul_node_124`). These are deterministic for a given
> MobiusBuilder export but can shift if the export graph changes; regenerate
> the config against the current export if node names move.

| Recipe | Pipeline | Output dir |
|---|---|---|
| `cpu/mixed/export.json` | `MobiusBuilder(fp32)` — export all components | `cpu/mixed/models` |
| `cpu/mixed/text.json` | `OnnxKQuantQuantization(int4, block=32)` + int8 upcast of sensitive weights — quantize decoder | `cpu/mixed/models/decoder` |
| `cpu/mixed/vision.json` | `OnnxBlockWiseRtnQuantization(int8, block=128)` — quantize vision encoder | `cpu/mixed/models/vision_encoder` |
| `cpu/mixed/audio.json` | `OnnxBlockWiseRtnQuantization(int8, block=128)` — quantize audio encoder | `cpu/mixed/models/audio_encoder` |
| `cpu/mixed/embedding.json` | `OnnxBlockWiseRtnQuantization(int8, block=128)` — quantize token embedding | `cpu/mixed/models/embedding` |
| `cuda/mixed/export.json` | `MobiusBuilder(fp16)` — export all components | `cuda/mixed/models` |
| `cuda/mixed/text.json` | `OnnxKQuantQuantization(int4, block=32)` + int8 upcast of sensitive weights — quantize decoder | `cuda/mixed/models/decoder` |
| `cuda/mixed/vision.json` | `OnnxBlockWiseRtnQuantization(int8, block=32)` — quantize vision encoder | `cuda/mixed/models/vision_encoder` |
| `cuda/mixed/audio.json` | `OnnxBlockWiseRtnQuantization(int8, block=32)` — quantize audio encoder | `cuda/mixed/models/audio_encoder` |
| `cuda/mixed/embedding.json` | `OnnxBlockWiseRtnQuantization(int8, block=32)` — quantize token embedding | `cuda/mixed/models/embedding` |

**Run order**: export first, then text, vision, audio, and embedding (the
latter four can run in parallel):

```bash
# CPU mixed
olive run --config cpu/mixed/export.json
olive run --config cpu/mixed/text.json
olive run --config cpu/mixed/vision.json
olive run --config cpu/mixed/audio.json
olive run --config cpu/mixed/embedding.json

# CUDA mixed
olive run --config cuda/mixed/export.json
olive run --config cuda/mixed/text.json
olive run --config cuda/mixed/vision.json
olive run --config cuda/mixed/audio.json
olive run --config cuda/mixed/embedding.json
```

K-Quant (Q4_K_M) is significantly faster with GPU acceleration —
install `cupy-cuda12x` for a 19–51× speedup during quantization.

## Build

For the latest optimized recipes, run the five stages in
[Optimized CPU INT4](#optimized-cpu-int4) or
[Optimized CUDA INT4](#optimized-cuda-int4). The commands below are the
alternative, single-stage full-precision and K-Quant builds.

```bash
# CPU, full precision
olive run --config cpu/fp32/config.json

# CPU, INT4 (K-Quant)
olive run --config cpu/int4/config.json

# CUDA, FP16
olive run --config cuda/fp16/config.json

# CUDA, INT4 (K-Quant)
olive run --config cuda/int4/config.json
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

Use `--model-path` for optimized packages; `--variant` does not accept
`int4-optimized`. The bundled helper provides text-only smoke tests: it
preserves the package's `past_present_share_buffer` setting, including the
shared buffers required by CUDA graphs. Interactive mode starts a fresh
generator for every prompt, so these smoke-test timings do not measure
persistent-cache continuation.

```bash
# CPU optimized INT4 (smoke test)
python inference.py --model-path cpu/int4-optimized/models --prompt "Hello"

# CUDA optimized INT4 (smoke test)
python inference.py --model-path cuda/int4-optimized/models --prompt "Explain quantum computing"

# Text-only (CPU, fp32)
python inference.py --prompt "What is the capital of France?"

# CPU INT4
python inference.py --variant int4 --prompt "Hello"

# CUDA INT4
python inference.py --device gpu --variant int4 --prompt "Explain quantum computing"

# CUDA mixed (mixed int4/int8 decoder + int8 vision/audio/embedding)
python inference.py --device gpu --variant mixed --prompt "Explain quantum computing"

# Interactive mode
python inference.py --device gpu --variant mixed --interactive
```

## Evaluation

### MMLU (text)

Run through Olive with the `LMEvaluator` configs in `eval/` (mixed model):

```bash
olive run --config eval/mmlu_cpu.json    # CPU
olive run --config eval/mmlu_cuda.json   # CUDA
```

The standalone script defaults to MMLU-Pro, not MMLU. It also supports `--task`,
`--limit`, and other variants:

```bash
# MMLU-Pro (default 100 samples), CPU
python eval.py

# CUDA INT4
python eval.py --device gpu --variant int4

# CUDA mixed
python eval.py --device gpu --variant mixed
```

### Vision — AI2D (exact_match)

> **Note**: The `olive run` eval configs require Olive to support nested model
> layouts in the evaluator's genai_config.json discovery. Until then, use
> a custom evaluation script.

### Audio — FLEURS ASR (WER)

> **Note**: The `olive run` audio eval configs require Olive to add gemma4
> as a supported model type in the speech evaluator. Until then, use a custom
> evaluation script.

## References

- Mobius docs: <https://github.com/onnxruntime/mobius>
- Olive `MobiusBuilder` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py>
- Olive `OnnxKQuantQuantization` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/kquant_quantization.py>
