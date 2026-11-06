# Qwen3-VL-2B-Instruct Mobius / ONNX Runtime GenAI Example

Export [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) with Olive's MobiusBuilder, quantize with Olive's native MLAS RTN pass, and run text and image inference with ONNX Runtime GenAI.

Mobius constructs all three connected components: vision encoder, embedding,
and decoder. CPU uses INT4 projections and embedding tables (decoder group32;
vision/embedding group128) with FP32 compute/KV. Normalization, convolution,
and rotary caches remain floating point. CUDA uses 131 INT4 and 66 INT8 decoder
projections, including an INT4 output head, with FP16 compute/KV,
vision, and embeddings.

DeepStack is explicitly disabled to match the tuned ModelBuilder baseline.
This is not Mobius's default Qwen3-VL behavior.

## Prerequisites

```bash
pip install -r requirements.txt
```

Use Olive and Mobius source builds containing `prune_prefill_prefix`,
`runtime_last_token_indices`, `disable_deepstack`, and `processor_config`
support in MobiusBuilder.
The review changes use `sunghcho/qwen-vl-cuda-pr-20261106` in the Mobius,
Olive, ONNX Runtime, and GenAI repositories. Install the matching source
checkouts when reviewing this branch.
The published `mobius-onnx==0.1.0` alone does not provide these options.
The RTN node-glob support is tracked separately in
[Olive #2531](https://github.com/microsoft/Olive/pull/2531) and must also be
available before running this recipe. That existing feature is intentionally
not duplicated by the MobiusBuilder integration changes.
The recipe requires ONNX IR 1.1.0 or newer; older 0.2.1 releases do not support
the `max_shard_size_bytes` argument used by the Mobius package writer.

```bash
# When Olive and Mobius are sibling checkouts under quant_kv:
pip install -e ../../../olive -e ../../../mobius
```

Install ONNX Runtime GenAI based on your target device:

| Device | Install Command |
|--------|-----------------|
| GPU (CUDA) | `pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |
| CPU | `pip install onnxruntime-genai --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |

The installed wheel must include the reserved `last_token_indices` runtime
support described below; otherwise use a matching source build or disable both
prefill-pruning options and re-export.

## Steps

### 1. Export & Optimize Models (CPU)

Run commands from this `builtin` directory. Each device has one declarative
Olive workflow: export the complete Mobius package once, then apply the
component-specific quantization passes. Runtime and native image-processor
settings are declared in the export pass, so no Python orchestration or
post-export configuration script is required.

| Command | Description |
|---------|-------------|
| `olive run --config cpu_and_mobile/config.json` | Full CPU export: decoder INT4/group32, vision and embedding INT4/group128 |
| `olive run --config cuda/config.json` | Full CUDA export: mixed INT8/INT4 decoder, FP16 vision and embedding |

Models and generated runtime configs are written to `<config-dir>/models`.

Both exports connect vision -> embedding -> decoder through `inputs_embeds`.
Component filenames remain `vision_encoder/model.onnx`, `embedding/model.onnx`,
and `decoder/model.onnx`; the package root contains the GenAI and processor
configuration and tokenizer files. Multi-image inference remains supported.

The CPU group128 pass explicitly selects learned position/token embedding
Gathers, including the learned position Gathers inside the vision Scan
subgraph. Decoder and vision rotary cosine/sine lookup tables remain floating
point; selecting all constant Gathers would incorrectly quantize these caches.

### CUDA recipe

The [CUDA workflow](cuda/config.json) uses symmetric group32 MLAS RTN
with the same 66 KLD-selected INT8 projections as the tuned baseline.
It quantizes those projections to INT8 first, then the remaining float
MatMuls to INT4. Module-path globs tolerate Mobius's numeric node-name suffixes;
already quantized nodes are not requantized.
Decoder filters select only projection MatMuls: rotary cosine/sine cache Gathers
must remain floating point. CUDA vision and embedding weights are not quantized.
About 82% of decoder-body parameters remain INT4, and the output head is INT4.
Vision, embeddings, and KV cache stay FP16.

The allocation was calibrated offline on checkpoint
`89644892e4d85e24eaac8bacfd4f463576704203` using WikiText-2 train and generated
math/instruction data. Export applies the fixed overrides without calibration
or GPTQ. Recalibrate when changing the checkpoint or precision budget.

Decoder CUDA graph capture is enabled; embedding and vision capture are
disabled. Re-export to apply recipe changes.

### Prefill runtime compatibility

CPU and CUDA exports use last-token-only prefill, returning `[batch, 1, vocab]`
to reduce compute and memory.
It requires an ONNX Runtime GenAI build with `last_token_indices` support.
The runtime supplies this reserved graph input automatically; it is deliberately
not listed in the GenAI JSON's decoder-input mapping.
Attention and KV generation still process the full prompt. CPU decoder
projections remain all INT4, with FP32 KV.

For older runtimes or tools requiring every prompt logit, set both
`prune_prefill_prefix` and `runtime_last_token_indices` to `false` in the
workflow's `export` pass and run a new export.
FP16 rounding can differ between full-prefill and last-token projections.
Olive's incremental ORT GenAI text-evaluation backend supports pruned logits
with batch size 1; MMLU does not require a separate full-logit export.

Arithmetic and strict-formatting failures remain; validate on your workload.

### 2. Run Inference

From the top-level model directory:

```bash
# Text inputs (CPU, default)
python inference.py --prompt "What is the capital of France?"

# Image inputs (CPU, default)
python inference.py --prompt "Describe this image" --image cat.jpeg

# Image inputs (CUDA)
python inference.py --model_path cuda/models --prompt "Describe this image" --image cat.jpeg

# Interactive mode
python inference.py --interactive
```

**Multi-image inference** is supported via `model-mm.py` from the `onnxruntime-genai` examples:

```bash
# Two images — compare or reason across multiple images
# Adjust paths to your onnxruntime-genai checkout and model directory
python <onnxruntime-genai>/examples/python/model-mm.py \
    -m <path-to-builtin>/cpu_and_mobile/models \
    -up "Are these two images the same?" \
    --image_paths image1.jpeg image2.jpeg \
    --non_interactive
```

## Evaluation

Evaluation uses Olive's built-in evaluators, not a custom scoring script.
See [evaluation configuration](eval/README.md) for the exact protocol.

```bash
cd eval
# First 100 AI2D test diagrams: built-in vision_vqa_pre_process + exact_match
olive run --config ai2d_cpu.json
olive run --config ai2d_cuda.json

# Five-shot MMLU: built-in LMEvaluator + incremental ORT GenAI backend
olive run --config mmlu_cpu.json
olive run --config mmlu_cuda.json
```

AI2D uses greedy generation with a 4,096-token total sequence limit.
MMLU's `limit: 100` applies **per subject**, not to the aggregate dataset.
For a quick MMLU smoke evaluation, use a copy with `limit: 1` (57 subjects).
Compare Mobius and baseline packages using identical built-in configurations.

### Earlier built-in smoke evaluation

Both exporters used the pinned checkpoint, the same native runtime
(ORT 1.32.0 / GenAI 0.18.0-dev), and identical evaluation configurations,
images, questions, targets, and generation limits. CPU used 24 EPYC 7V12
cores on one NUMA node; CUDA used an A100 80 GB.

| Device | Exporter | AI2D first 100 | Five-shot MMLU smoke, 57 questions |
|--------|----------|---------------|-----------------------------------|
| CPU | Tuned ModelBuilder baseline | 67/100 (67.00%) | 35/57 (61.40%) |
| CPU | Mobius, declarative pipeline | 68/100 (68.00%) | 35/57 (61.40%) |
| CUDA | Tuned ModelBuilder baseline | 66/100 (66.00%) | 35/57 (61.40%) |
| CUDA | Mobius, declarative pipeline | 66/100 (66.00%) | 35/57 (61.40%) |

No samples were skipped. The MMLU validation used `limit: 1` per subject,
not the public configs' `limit: 100`; it is not a full MMLU benchmark.
There is no measured aggregate regression in these checks; a one-question
AI2D difference is not evidence of a statistically significant improvement.

Packed-weight/scale values, bit allocation, and group sizes match the respective
baselines. Text, image, ordered two-image, and pixel-only logit controls passed.
This does not establish bit-identical graphs or numerical outputs: Mobius's CUDA
vision graph accepts FP32 pixels and casts them to FP16 internally, whereas the
baseline exposes FP16 pixels.
Metrics and per-sample predictions come from Olive's built-in evaluators;
evaluation wall time is not a prefill/decode throughput measurement.

## Final CUDA performance and quality gate

The clean-build/export run `pr-final-gate-20261105-v1` measured the combined
Mobius/Olive/ORT/GenAI feature stack through Foundry Local against a freshly
built llama.cpp. These are pinned pre-PR measurements, not isolated speedups
attributable to each individual PR or measurements of subsequent upstream
rebases. Mixed native INT4/INT8 group32 and GGUF Q4_K_M are different quantizers.

Both engines used one A100-SXM4-80GB, eight CPU threads, batch 1, an 8,192-token
runtime context, FP16 KV, 2,048-token prefill chunks, identical rendered prompts
and input IDs, and exactly 128 generated tokens without prefix-cache hits.
Two reversed process-order blocks each used two warmups and seven measured
requests per length: 14 measurements per cell. Full multimodal packages were
loaded, but performance and memory workloads were text-only.

Median throughput, tokens/second:

| Prompt tokens | llama.cpp decode | Mobius decode | Decode gain | llama.cpp effective prefill | Mobius effective prefill | Prefill gain |
|---------------|------------------|---------------|-------------|-----------------------------|--------------------------|--------------|
| 128 | 301.34 | 349.41 | +15.95% | 5,819 | 11,925 | +104.92% |
| 512 | 297.92 | 340.51 | +14.29% | 12,130 | 28,994 | +139.03% |
| 2,048 | 287.28 | 325.15 | +13.18% | 16,272 | 42,863 | +163.42% |
| 4,096 | 271.73 | 304.94 | +12.22% | 17,245 | 40,790 | +136.53% |
| 7,680 | 255.06 | 288.68 | +13.18% | 17,044 | 37,264 | +118.64% |

Geometric-mean decode gain is **13.76%**. Effective prefill is prompt tokens
divided by HTTP time to first content token, including host and first-token
overhead; it is not isolated kernel throughput. Decode uses the 127 intervals
between 128 visible content tokens.

Full matched evaluation used all 14,042 MMLU and 3,088 AI2D examples, paired by
sample identity, targets, prompt/input token hashes, image hashes, and counts:

| Model | MMLU correct | MMLU accuracy | AI2D correct | AI2D accuracy |
|-------|--------------|---------------|--------------|---------------|
| Mobius mixed INT4/INT8 | 8,389 | 59.7422% | 2,170 | 70.2720% |
| Native FP16 control | 8,513 | 60.6253% | 2,229 | 72.1826% |
| llama.cpp Q4_K_M | 8,349 | 59.4573% | 2,272 | 73.5751% |
| llama.cpp F16 control | 8,515 | 60.6395% | 2,315 | 74.9676% |

The measured native vision path omits DeepStack intermediate vision-feature
injection, unlike llama.cpp. This missing feature is a known architectural
limitation when interpreting the AI2D gap, not evidence that upstream Mobius
lacks DeepStack support. The recipe explicitly disables its existing support.
Separately, Mobius loses nine AI2D answers (0.2915 percentage points) versus
the accepted native recipe, which also disabled DeepStack; that additional
regression is not explained by the shared DeepStack omission. MMLU is unchanged
against that accepted native recipe. Quality therefore remains **review required**,
not an unconditional default-model promotion.

Separate per-process NVML trials sampled every 20 ms:

| Engine | Loaded VRAM MiB | Sampled peak VRAM MiB | Peak host RSS MiB |
|--------|-----------------|-----------------------|-------------------|
| llama.cpp Q4_K_M | 3,802 | 3,822 | 1,392.26 |
| Mobius | 2,856 | 4,272 | 815.64 |

Mobius has lower loaded VRAM but a 450 MiB higher sampled peak. Sampling can
miss instantaneous peaks; these are not worst-case vision-memory measurements.

The gate used ORT 1.32.0 / GenAI 0.18.0-dev, `ORT_FPA_INTB_GEMM=1`,
`ORT_ENABLE_SINGLE_WARP_INT4_GEMV=1`, a frozen load-only A100 tactic cache, and
`session.use_device_allocator_for_initializers=1`. The generic GEMV control
selects FP16, symmetric group32 INT4, one input row, no explicit zero points
or fused bias, and `(N, K)` of `(1024, 2048)`, `(2048, 2048)`, or `(2048, 6144)`.
Other shapes keep existing dispatch.

### Earlier CPU control

The earlier CPU run used 24 physical EPYC 7V12 cores on one NUMA node, FP32 KV,
and six measured requests per length. It was not repeated by the final CUDA gate:

| Prompt tokens | Mobius CPU decode | llama.cpp CPU decode |
|---------------|-------------------|-----------------------|
| 128 | 38.12 | 43.61 |
| 512 | 36.99 | 36.67 |
| 2,048 | 32.44 | 24.53 |
| 4,096 | 26.70 | 16.88 |
| 7,680 | 19.83 | 10.79 |

The CPU geometric-mean decode ratio was 1.276x, but short-prompt decode and
overall effective prefill remained slower. This is separate evidence, not a
claim that the final CUDA gate also revalidated CPU throughput.

## Directory Structure

```
builtin/
  inference.py             # Text and image inference
  eval/
    ai2d_cpu.json           # Built-in vision evaluator
    ai2d_cuda.json
    mmlu_cpu.json           # Built-in text evaluator
    mmlu_cuda.json
  cpu_and_mobile/
    config.json            # Mobius export -> decoder INT4 -> vision/embedding INT4
    models/                # Generated package
  cuda/
    config.json            # Mobius export -> selected decoder INT8 -> remaining INT4
    models/                # Generated package
```
