# Private INT3 Model: Paged ONNX Runtime GenAI Engine CUDA Performance

The benchmark model is private. Model names, original artifact filenames,
local directories, and machine identifiers are omitted or anonymized.
`private_paged_int3_model` and `private-model-paged-engine-20261106` are report
aliases, not actual filenames or model-family attributions. Measurements and
software/model hashes are retained unchanged. Public software links do not
identify the model's owner or source-checkpoint provenance.

Measurements completed on **2026-11-06**, from 17:39:00 to 17:42:32 UTC.
This is a separate report, not an update to the historical
[private-model CUDA performance report](https://github.com/microsoft/olive-recipes/blob/825297bd3a5e5d8d88513f32b9df5a84ff241d41/PRIVATE_MODEL_CUDA_PERFORMANCE.md).
Its measurement definitions inform this evaluation, but the invocation,
hardware, cache management, and output processing differ as detailed below.
No controlled speedup against that report is claimed.

## Scope

The supplied, unchanged paged-attention graph was evaluated through its
matching ONNX Runtime GenAI Engine API. No graph rewrite, weight change,
runtime rebuild, or speculative draft model was used. This is a synthetic
batch-1 performance evaluation, not an accuracy, perplexity, representative
production-routing, or natural-language quality benchmark.

## Runtime and Hardware

| Item | Value |
| --- | --- |
| GPU | NVIDIA H200, one isolated compute device |
| Driver | 580.159.04 |
| Python | 3.12.15 |
| ONNX Runtime | 1.32.0, supplied local wheel |
| ONNX Runtime GenAI | 0.20.0-dev, matching local wheel |
| ORT build | CUDA 13.0, SM90, Release; FP8 QMoE build support enabled |
| GenAI build | Wheel produced by the supplied Debug build |
| Execution environment | Dedicated host Python virtual environment; no Docker |

FP8 build support does not establish FP8 execution for this INT3 model.
The three mapped ORT libraries were SHA256-checked against the supplied build
output, rather than identified by package version alone. All eight formal
workers mapped identical runtime libraries and observed the same GPU UUID;
the UUID is retained locally, not published. A current source checkout is
not sufficient to establish the historical build's source revision or
cleanliness, so neither is claimed here.

| Mapped Library | SHA256 |
| --- | --- |
| libonnxruntime.so.1.32.0 | `d5b34b5fe1ccab8519a6288b16e26a0b523715c27bd445a9f7c8882891ffe298` |
| libonnxruntime_providers_cuda.so | `1ae23b6905b380a54e18ebe64a175601df50b4ae52cd3464fe26e7a74ab96bcf` |
| libonnxruntime_providers_shared.so | `17f7cf47ad0d7b5ac895ae588fd62c7f85a13842588161b6a24c7d480f062be4` |
| onnxruntime_pybind11_state.so | `c431352452de87ba3e5203ea05f886935095ca864c91675251ef95be04a32850` |
| libonnxruntime-genai-cuda.so | `c68d230a4cb815bf2b72bc439e3dbe58a32ce6443febf5d25f10da8032dbe5d6` |
| libonnxruntime-genai.so | `2ba6202cab9356c0675fe1d96f641f63c5d3cee622b1570c22ea32c4d5649a82` |
| onnxruntime_genai.cpython-312-x86_64-linux-gnu.so | `d6ca4746d6014910030a146253784f05c966eeda429dde889b9d3ffe3d10fb17` |

## Model Artifacts

Filenames in this table are anonymized aliases.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| private_paged_int3_model.onnx | 490793 | `d390742ab836bab75bf7a9dcac10eae5560468fab8a2d2afa5205492f2c60706` |
| private_paged_int3_model.data | 70574209280 | `6bcdb247f1b01574e3e9fdd97fe0d989f658d8303ebce7b41f15844093c51937` |
| Engine configuration | 2267 | `21a88b80d5b4448e4139248985af8270dcef316be22eac066fd12a5328e0a435` |

Artifacts were hashed before the matrix; size and nanosecond modification
time were unchanged afterward. The graph contains 42 PagedAttention,
21 QMoE, 295 MatMul, and 253 SimplifiedLayerNormalization nodes. These are
graph metadata counts, not executed CUDA operator or kernel counts.
INT3 describes expert quantization, not uniform three-bit model precision.

## Measurement Method

- Batch size 1; input lengths 128, 512, 2048, and 4096.
- Each request produces 128 predictions: the prefill prediction and
  127 decode transitions.
- Round 1 runs lengths ascending; round 2 descending. Eight fresh sequential
  processes each perform 3 warmup and 5 measured requests. Each table row
  pools 10 requests. Totals are 40 measured requests, 5120 predictions,
  and 5080 decode transitions; 24 warmups and all preflight probes are excluded.
- Pretokenized INT64 prompts follow
  `[100 + (i % 128) for i in range(input_length)]`. Tokenization, prompt
  construction, initialization, and request-object creation are outside TTFT.
- Natural greedy decoding uses minimum and maximum generated lengths of 128
  to suppress early EOS stopping. This is not the reference's unrestricted
  argmax with EOS ignored, or its shared teacher-forced continuation.
- CUDA graphs and timing-time ORT profiling are disabled. Intra/inter-op
  threads are 8/1; QMoE scratch is 67108864 bytes per operation. Dense
  auto-tuning and opt-in BF16 norm/cast fusion are disabled;
  GroupQueryAttentionPreNormFusion is explicitly excluded.
- Original paged-cache and sliding-window configuration is retained. Prefix
  caching is disabled. Each request has a new request object, but the Engine
  reuses its preallocated KV pool; explicit zeroing of all cache bytes is not
  performed.
- Engine max batch size is overridden to 1. Block size 256, max scheduled
  tokens 2048, GPU utilization factor 0.6, and search chunk size 256 are retained.
- Every `engine.run` call is followed by `cudaDeviceSynchronize`. TTFT spans
  `begin_turn` through first-token event handling, including scheduling,
  binding/input/output work, GPU inference/token selection, and synchronization.
- Decode wall time spans the first to last token event, including 127 subsequent
  synchronized Engine calls and Python event handling. Token hashing is outside
  timing. Request latency is TTFT plus decode wall time; cleanup is excluded.
- End-to-end prefill throughput is pooled input tokens / sum(TTFT).
  Engine-run prefill throughput is pooled input tokens / sum(synchronized
  `engine.run` times through first token). Neither is pure ORT-run throughput.
- Throughputs use ratios of pooled totals, not averages of per-request rates.
  p50/p95 use NumPy linear interpolation; values are rounded only for display.
- Initialization includes Model and Engine construction and final synchronization,
  including weight loading/prepacking and KV-pool allocation. It is not a
  prepack-only or guaranteed cold-filesystem measurement.
- GPU compute processes are checked idle before each worker. NVML samples the
  measured interval with a requested 5 ms period and rejects additional compute
  PIDs. Clocks/power are not locked; unsampled interference and transient peaks
  cannot be excluded. Other GPUs host unrelated workloads, which may affect
  shared host resources.

## Pooled Results

Memory is NVML-sampled process residency, including the preallocated KV pool,
not minimum model footprint or exact allocation peak.

| Input Tokens | Requests | TTFT p50 ms | TTFT p95 ms | E2E Prefill tok/s | Engine-Run Prefill tok/s | Decode tok/s | Peak Process GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10 | 71.825 | 71.900 | 1781.917 | 1786.686 | 108.863 | 122.914 |
| 512 | 10 | 272.566 | 273.124 | 1878.886 | 1880.561 | 102.439 | 123.072 |
| 2048 | 10 | 1136.165 | 1138.652 | 1802.794 | 1803.751 | 104.484 | 123.072 |
| 4096 | 10 | 2275.747 | 2279.797 | 1799.196 | 1799.997 | 103.385 | 123.072 |

| Input Tokens | Request p50 ms | Request p95 ms | Engine Calls to First Token | Distinct Output Streams / 10 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1238.477 | 1240.098 | 1 | 1 |
| 512 | 1512.055 | 1515.687 | 2 | 1 |
| 2048 | 2351.484 | 2353.740 | 8 | 1 |
| 4096 | 3505.823 | 3517.876 | 16 | 1 |

Observed calls to first token are consistent with 256-token chunked prefill,
not unchunked prefill or 2048-token chunks. A call may contain internal work
not exposed by the public API.

## Per-Process Initialization and Sampling

| Round | Input | Decode tok/s | Initialization s | Peak Process GiB | NVML Samples | Median Gap ms | Max Gap ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 108.812 | 4.186 | 122.914 | 1195 | 5.182 | 6.237 |
| 1 | 512 | 102.353 | 4.179 | 123.072 | 1458 | 5.187 | 5.446 |
| 1 | 2048 | 104.576 | 4.356 | 123.072 | 2270 | 5.182 | 5.539 |
| 1 | 4096 | 104.225 | 4.215 | 123.072 | 3371 | 5.182 | 5.587 |
| 2 | 4096 | 102.558 | 4.449 | 123.072 | 3393 | 5.182 | 5.841 |
| 2 | 2048 | 104.392 | 4.279 | 123.072 | 2269 | 5.181 | 5.945 |
| 2 | 512 | 102.526 | 4.322 | 123.072 | 1456 | 5.190 | 5.957 |
| 2 | 128 | 108.913 | 4.475 | 122.914 | 1196 | 5.179 | 5.545 |

## Validation and Comparison Limits

All 40 measured requests completed with the expected prediction count and
selected tokens within vocabulary range. At each input length, all ten
requests produced an identical token stream. Runtime/model identity and
sampled GPU compute-process isolation checks passed. Three focused harness
tests passed, checking pooled-total arithmetic, linear percentiles, and
rejection of mixed prediction counts.

Public Engine events expose selected token IDs, not full logits or a pure
ORT-run timer. No all-logits finite check, cross-runtime allclose, forced-input
equivalence, or task accuracy is claimed. Token-stream repeatability is not
numerical correctness or quality validation. No separate profiler capture
was collected: graph metadata and CUDA provider residency do not establish
per-node placement, INT3 fast-path dispatch, or particular CUDA kernel choices.

The referenced report uses A100-SXM4-80GB, a GQA graph and different runtime,
fixed-capacity shared KV buffers, a common forced continuation, and CPU
last-logits finite checking, argmax, and hashing. This evaluation uses H200,
paged attention, an Engine KV pool, natural greedy per-input continuations,
GPU token selection, and chunked prefill. Performance ratios between these
reports would confound those changes and must not be treated as controlled
speedups. Two rounds provide a limited sample without confidence intervals.

## All-Methods Combined Experiment (2026-11-06)

This follow-up combines all five requested optimization ideas in one separate
experimental graph. The original graph and weights are not overwritten.
Logit differences are not an acceptance gate for this performance experiment;
the candidate is not accuracy-qualified or recommended for production.
The historical baseline tables above remain unchanged.

### Combined Graph

- Remove 168 standalone Q/K norm Cast nodes using the existing BF16 norm
  export, before quantization and attention-internal norm fusion.
- Apply symmetric block-32 weight-only quantization: 273 ordinary projection
  MatMuls to INT4, 21 router MatMuls and the FP32 head to INT8, and the FP32
  embedding to INT8 GatherBlockQuantized. Activations remain BF16/FP32.
- Fuse 84 FP32 residual Add + RMSNorm pairs into
  SkipSimplifiedLayerNormalization, retaining residual sums through output 3.
- Merge Q/K/V INT4 projection weights in each of 42 layers, producing widths
  8192 from 6144/1024/1024, with packed weights and BF16 scales concatenated
  byte-for-byte along output channels.
- Move 84 per-head Q/K RMSNorms into PagedAttention inputs 12/13 with the
  original epsilon, approximately 1e-5. Feed merged raw QKV directly through
  the existing packed-input reshape: no intermediate Split/Concat is required
  in this combined graph. Both rotary and non-rotary layers are included.
- Merge 21 equal-width dense gate/up projection pairs into width 7168 and
  feed single-input SwiGLU with activation_alpha=1 and round_activation=1.
  Stored half ordering follows the actual Sigmoid/Mul topology, not names.

All 21 INT3 QMoE nodes and their protected initializers match the original
serialized metadata. The retained source weight file has the original SHA256.
All 126 newly merged weight/scale tensors were independently byte-verified
against their sources; no additional requantization is done during merging.
All 42 paged-cache and metadata input/output interfaces and existing attention
attributes are preserved, with only optional Q/K scale inputs and epsilon added.

The combined graph SHA256 is
`116e601080a171a21b1e9c9f6553bd68efca0766b3c70cb15085ae092252cf56`.
Its added merged-weight sidecar is 997097472 bytes. It also requires the mixed
quantization sidecar and retained original weights; it is not self-contained
in just the graph and newly merged sidecar. Original unused initializers have
not been compacted out of the retained sidecars.

| Raw Graph Operator | Original | Combined |
| --- | ---: | ---: |
| MatMul | 295 | 0 |
| MatMulNBits | 0 | 190 |
| GatherBlockQuantized | 0 | 1 |
| Add | 84 | 0 |
| SimplifiedLayerNormalization | 253 | 85 |
| SkipSimplifiedLayerNormalization | 0 | 84 |
| Cast | 420 | 210 |
| Reshape | 273 | 105 |
| Sigmoid | 21 | 0 |
| Mul | 42 | 0 |
| SwiGLU | 0 | 21 |
| PagedAttention | 42 | 42 |
| QMoE | 21 | 21 |

These are exported graph counts, not profiler-verified executed operators or
CUDA kernel launches. Some original identity Casts may already be runtime-eliminated.

### Fresh Original-versus-Combined A/B

Both conditions use the same supplied runtime libraries, H200, synthetic
prompts, natural greedy generation, CUDA-graph setting, 8/1 threads, 64 MiB
QMoE scratch, and synchronization/timing definitions from this report.
The original Engine cache-pool configuration and 256-token prefill chunks are
retained. Two reversed-order rounds compare adjacent conditions at each input
length, with 3 warmups and 5 measured requests per fresh process. Sixteen
processes give 80 measured requests and 10160 decode transitions, excluding
48 warmups, all export tests, and separate preflight requests.

Every worker passed GPU idle/sampled compute-PID isolation, library identity,
output count/range, and configuration checks. Aggregates were independently
recomputed from all worker JSON. Clocks/power remain unlocked.

| Input | Original Decode tok/s | Combined Decode tok/s | Gain | Original TTFT p50 ms | Combined TTFT p50 ms | TTFT Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 110.625 | 151.473 | +36.93% | 71.630 | 73.329 | +2.37% |
| 512 | 103.453 | 137.454 | +32.87% | 272.405 | 275.402 | +1.10% |
| 2048 | 102.524 | 138.734 | +35.32% | 1137.697 | 1146.273 | +0.75% |
| 4096 | 104.296 | 137.233 | +31.58% | 2277.858 | 2303.947 | +1.15% |

| Input | Original / Combined Request p50 ms | Original / Combined Peak Process GiB | Original / Combined E2E Prefill tok/s |
| ---: | --- | --- | --- |
| 128 | 1221.039 / 911.898 | 122.914 / 115.285 | 1786.874 / 1744.567 |
| 512 | 1499.736 / 1199.071 | 123.072 / 115.482 | 1880.476 / 1860.475 |
| 2048 | 2374.872 / 2061.489 | 123.072 / 115.482 | 1800.229 / 1785.896 |
| 4096 | 3495.602 / 3230.862 | 123.072 / 115.482 | 1798.168 / 1777.166 |

| Input | Round 1 Original / Combined Decode tok/s | Round 2 Original / Combined Decode tok/s |
| ---: | --- | --- |
| 128 | 111.173 / 151.681 | 110.081 / 151.266 |
| 512 | 104.542 / 137.222 | 102.387 / 137.687 |
| 2048 | 102.889 / 138.325 | 102.161 / 139.145 |
| 4096 | 103.789 / 136.136 | 104.808 / 138.349 |

Both rounds improve decode at all lengths, with slightly worse TTFT.
Process residency drops approximately 7.59-7.63 GiB, still including the
preallocated KV pool. These gains are measured for the whole combination,
not sums of historical percentages or attribution to individual fusions.

### Numerical Limits

Small CUDA checks passed for INT4/INT8 MatMul with FP32/BF16 inputs,
INT8 embedding, FP32 residual fusion, and packed SwiGLU. Packed versus
two-input SwiGLU outputs matched bitwise for FP32/BF16 at 1/128/256 rows;
checked outputs were finite. The full combined model completed a separate
128-token CUDA preflight and all formal requests, but no independent
PagedAttention Q/K norm numerical matrix or full-vocabulary logits probe
was collected for this combination.

Each condition has one repeatable output stream across all ten requests at
each length. A first-request natural greedy comparison has 127/128,
26/128, 128/128, and 128/128 position-wise token matches at inputs
128/512/2048/4096 respectively. These are independent continuations, not
matched teacher-forced states, so divergence changes subsequent tokens and
potential MoE routing. Position matches are not accuracy or logits-error
metrics. Even identical tokens do not establish identical logits.

No full-logits finite/allclose result, quantization-quality metric, perplexity,
or task accuracy is claimed. This experiment establishes observed performance
of a lossy combined candidate, not equivalent arithmetic or production
readiness. The local alias `private-model-paged-combined-20261106` covers the
combined exporter/tests, candidate manifests, weight-byte verification,
preflight, all 16 worker records, and validated summary; these remain private.

## Retained Artifacts

The local report alias `private-model-paged-engine-20261106` covers the
worker JSON and logs, unrounded per-request timings, prompt/output hashes,
model/runtime manifests, environment/dependency records, preflight results,
summary, and local benchmark/orchestration scripts. Preflight results are
not pooled into the formal matrix. These private/local artifacts have no
public download link and are not included in this report-only submission.
This is not a public self-contained reproduction package. No model, weights,
tokenizer, raw logits, local paths, or machine UUIDs are published.