# Private INT3 Model: ONNX Runtime CUDA Performance

The benchmark model is private. Its name, original artifact filenames, and
local artifact-directory identifiers are anonymized in this report.
`private_int3_model` and `private-model-*` are report aliases, not actual
filenames or a model-family attribution. Measurements and hashes are retained
unchanged. Links to public runtime and recipe repositories identify software
sources only, not the model's owner or provenance.

Short-input measurements were completed on 2026-10-27; the long-input extension
below was completed on 2026-10-28, and initial decode profiling on 2026-10-29.
The original measurements use one ORT build and one GPU. Later sections
compare ORT revisions and execution experiments on the same supplied
`private_int3_model` artifact, not another inference engine or model.

**Latest measured performance (2026-11-03):** the
[residual Add + RMSNorm fusion](#residual-add--rmsnorm-fusion-on-latest-main-2026-11-03)
reaches **87.028 / 83.366 decode tok/s** at input lengths 128/512,
**+4.29% / +3.36%** versus the mixed-quantized baseline in a fresh A/B
using the same clean latest-main build. Prefill is essentially unchanged.
**This fusion is not lossless:** relative logit L2 differences are
**3.66% / 3.62%** versus the mixed-quantized baseline; task accuracy remains
unvalidated. The original and baseline artifacts are retained unchanged.

**Earlier mixed-quantization measurements:** the
[mixed INT4/INT8 export](#mixed-int4int8-weight-only-export-2026-11-03)
reaches **84.045 / 81.225 decode tok/s** at input lengths 128/512,
**+22.03% / +22.22%** versus the Q/K-export baseline in the same fresh A/B.
Prefill latency increases **7.66% / 1.66%**. This is lossy quantization,
not the earlier lossless norm/Cast optimization; numerical changes and
validation limits are reported with the results.

**Earlier follow-ups:** see the
[upstream main comparison](#latest-upstream-main-comparison-2026-11-01)
and [refreshed decode profiling](#refreshed-upstream-decode-profiling-2026-11-01),
followed by the
[equivalent norm/cast experiment](#equivalent-normcast-fusion-experiment-2026-11-01)
and the [Python-only Q/K export on unmodified ORT](#python-only-qk-norm-export-on-unmodified-ort-2026-11-02).
The original runtime identity and timing tables below remain historical
results; they are not measurements of the newer build.

The measurement definitions follow the
[Qwen3 INT3 performance report at the pinned revision](https://github.com/microsoft/olive-recipes/blob/264f863e6801e6b9ec1807f596515beeba4091ae/Qwen-Qwen3-30B-A3B/cuda/rtn_int3_fp16/PERFORMANCE.md).
**This is a different model, not an export of the Qwen recipe in this PR.**
Its tokenizer and source-checkpoint identity were not supplied. Inputs are
explicitly synthetic token IDs, so these results do not characterize natural
language generation quality, representative production routing, or accuracy.

## Runtime and Build Identity

| Item | Value |
| --- | --- |
| ORT source | `55ab2dfff389902bb98e3404276dc44e2c71b2ee` (`origin/main` at build time) |
| Installed package | `onnxruntime-gpu 1.32.0`, locally built wheel |
| Wheel | `onnxruntime_gpu-1.32.0-cp312-cp312-linux_x86_64.whl` |
| Wheel SHA256 | `8d5af66164fc41cf86720c24c3fb6a43da28758760b21942c694d8eacb99e572` |
| GPU | NVIDIA A100-SXM4-80GB, physical GPU 0 |
| GPU UUID | `GPU-739515b2-62b4-f900-86a2-50f55272a810` |
| Driver | `580.159.04` |
| Platform | Linux x86-64, `jiafa-dev` container |
| Configuration | Release, Ninja, CUDA 13.0, SM80 |
| Compiler | GCC/G++ 13.3.0; NVCC V13.0.88 |
| cuDNN | 9.23.2 |
| Python | 3.12.3 |
| CMake / Ninja | 4.4.3 / 1.13.0.git.kitware.jobserver-pipe-1 |
| Python dependencies | ONNX 1.22.0, NumPy 2.5.0, nvidia-ml-py 13.615.71 |
| Release C/C++/CUDA flags | `-O3 -DNDEBUG` |
| Build options | Shared library and Python wheel enabled; LTO and QUICK_BUILD disabled |

This was a clean build in a new output directory, not an incremental rebuild.
Submodules were synchronized to the ORT commit; ONNX source was the pinned
`e1ac8e74b620bbdc7ba07f0440806bd93f651a28`, without the dependency override
used by the historical Qwen comparison.
The installed wheel replaced the previous ORT package in the test virtual
environment. Timing processes imported that installed package, not the source
tree or a staged package. The mapped ORT library hashes were identical across
all eight timing processes:

| Mapped library | SHA256 |
| --- | --- |
| `onnxruntime_pybind11_state.so` | `b78d6f490990ae2dac81c33b0dfa9cf223654ece6d7bedfe03bf0449ed58b488` |
| `libonnxruntime_providers_shared.so` | `17f7cf47ad0d7b5ac895ae588fd62c7f85a13842588161b6a24c7d480f062be4` |
| `libonnxruntime_providers_cuda.so` | `9e30b8b508ff8126caebf2602755a73d8c434629955168894968d1349a0ea7d7` |

The GPU was checked idle before every worker. NVML sampling rejected an
additional or changed observed compute PID. Unsampled activity cannot be
excluded. Power limits and clocks were not locked by this experiment. A
during-run snapshot reported a 400 W power limit, 1410 MHz SM clock, and
1593 MHz memory clock; these are not guarantees of constant clocks.

## Model Artifact

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `private_int3_model.onnx` | 630457 | `91dc2b0031aae23bea17c7e1be9c7713cf290b7c931f0a60972bb1d4bd440a5c` |
| `private_int3_model.data` | 70574209280 | `6bcdb247f1b01574e3e9fdd97fe0d989f658d8303ebce7b41f15844093c51937` |

Graph plus external weights total **70574839737 bytes = 70.574839737 GB =
65.727941447 GiB**. Both files were hashed before the matrix, and their sizes
and modification times were checked unchanged afterward.

The graph has 42 GroupQueryAttention nodes and 21 QMoE nodes. QMoE metadata
specifies INT3 expert weights, group size 32, 512 experts, top-8 normalized
routing, fused SwiGLU, `swiglu_limit=100`, and `weights_prepacked=0`.
Expert scales and KV caches are BF16; logits are FP32 with vocabulary size
201088. Other BF16/FP32 tensors remain, including a float embedding, so this is
**not a uniformly three-bit model**.

Representative expert tensors:

| Tensor | Shape | Type |
| --- | --- | --- |
| FC1 packed weights | `[512,5376,576]` | UINT8 |
| FC1 scales | `[512,5376,48]` | BF16 |
| FC2 packed weights | `[512,1536,1008]` | UINT8 |
| FC2 scales | `[512,1536,84]` | BF16 |

These are graph metadata observations, not a verification of packed weight
contents, source checkpoint provenance, or quantization quality.

## Measurement Method

- Batch size 1; input lengths **128, 512, 2048, 4096**.
- Each request makes **128 predictions: one prefill prediction and 127 decode
  transitions**. EOS is ignored.
- Each input prompt is the INT64 array
  `[[100 + (i % 128) for i in range(input_length)]]`. There is no tokenization
  or text preprocessing in the timer.
- A separate natural-greedy 128-prediction reference at input length 128
  supplies the forced continuation for all timing requests, at all lengths.
  At decode transition `i`, the input is reference prediction `i - 1`.
  Longer-context continuations are synthetic teacher-forced workloads, not
  independent natural-greedy generations.
- Two rounds use fresh processes: lengths increase in round 1 and decrease in
  round 2. There are **8 sequential timing processes**, each with **3 warmup
  requests and 5 measured requests**. Each result row pools 10 requests.
- Totals are **40 measured requests, 5120 predictions, and 5080 decode
  transitions**. The **24 warmup requests**, reference request, preflight
  attempts, and separate dispatch captures are excluded from timing results.
- Intra/inter-op thread counts are **8/1**. CUDA graphs, ORT profiling, and
  QMoE debug/environment overrides are disabled for timing.
- CUDA is primary and required to load; Python provider fallback is disabled.
  CPU shape operations remain allowed, as in the referenced methodology.
- QMoE scratch configuration is
  `ep.cuda.qmoe_int_dequant_max_scratch_bytes=67108864` (64 MiB per operation,
  not a process-memory cap).
- Each request starts with freshly zeroed, synchronized BF16 GPU KV buffers
  and logical past length zero. The exported model requires in-place
  past/present sharing for sliding-window attention. Its 35 local-attention
  layers use capacity **511**, matching the exported `local_window_size`;
  the 7 global-attention layers use capacity **input length + 127**.
  Buffers are reset outside TTFT. This fixed-capacity cache interface differs
  from the historical Qwen graph.
- The model input named `past_seq_len` is populated with **total length - 1**,
  following GroupQueryAttention's input semantics, despite that input name.

Metric definitions:

- **TTFT:** pretokenized prompt to CPU argmax, including binding work,
  synchronized prefill, final-position logits D2H copy, all-value finite check,
  and argmax. Session initialization and KV reset are excluded.
- **Prefill throughput:** total input tokens / sum of synchronized ORT prefill
  run times. It is not input length divided by TTFT. Full-sequence logits
  stay on GPU; only the last-position vector is copied to CPU.
- **Decode throughput:** total 127-step transitions / sum of decode wall
  times, including Python, binding, sequence-length inputs, GPU execution,
  final-position D2H, finite checks, argmax, and logits hashing.
- **Request latency:** TTFT plus the 127-transition decode wall time.
- **Memory:** NVML-sampled process residency over measured requests, requested
  sampling interval 5 ms. The pooled table reports the larger process peak
  from the two rounds. It is not an exact allocation peak or device-wide
  usage; short-lived peaks can be missed.
- **Initialization:** complete ORT session construction, including graph work,
  loading, allocation, and prepacking. It is not a prepack-only timer or a
  guaranteed cold-filesystem measurement.

Throughputs use ratios of pooled totals, not averages of per-request rates.
p50/p95 use NumPy's default linear percentile interpolation. Values below are
rounded only for display.

## Pooled Results

| Input tokens | Measured requests | TTFT p50 ms | TTFT p95 ms | Prefill tok/s | Decode tok/s | Peak process GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10 | 272.310 | 273.239 | 471.131 | 60.196 | 69.561 |
| 512 | 10 | 1059.711 | 1060.865 | 483.535 | 58.459 | 71.686 |
| 2048 | 10 | 4225.346 | 4225.865 | 484.816 | 58.483 | 72.250 |
| 4096 | 10 | 8446.752 | 8447.184 | 484.983 | 57.695 | 74.945 |

| Input tokens | Init min s | Init max s | Request p50 s | Request p95 s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 7.382 | 7.533 | 2.383 | 2.392 |
| 512 | 7.564 | 9.089 | 3.232 | 3.236 |
| 2048 | 7.515 | 9.116 | 6.397 | 6.406 |
| 4096 | 7.398 | 7.461 | 10.648 | 10.651 |

## Per-Process Initialization and Memory Sampling

There are five measured requests per process. Memory samples include the
measured-request interval, not initialization or warmups. Process peaks vary
between rounds; no allocator-level cause is assigned to that variation.

| Round | Input | Init s | Peak process GiB | Samples | Interval p50 ms | Interval max ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 7.533 | 69.061 | 2267 | 5.259 | 7.194 |
| 2 | 128 | 7.382 | 69.561 | 2251 | 5.259 | 6.067 |
| 1 | 512 | 9.089 | 71.686 | 3063 | 5.260 | 10.204 |
| 2 | 512 | 7.564 | 71.686 | 3067 | 5.259 | 6.235 |
| 1 | 2048 | 9.116 | 72.250 | 6071 | 5.260 | 6.975 |
| 2 | 2048 | 7.515 | 70.250 | 6062 | 5.258 | 5.768 |
| 1 | 4096 | 7.461 | 73.186 | 10111 | 5.256 | 6.958 |
| 2 | 4096 | 7.398 | 74.945 | 10096 | 5.259 | 6.870 |

## Input and Repeatability Checks

Prompt hashes are SHA256 of the INT64 array's contiguous bytes. The forced
continuation hash is SHA256 of the 128 reference token IDs as INT64 bytes.
Both rounds matched their prompt hashes, all eight processes matched the
forced-continuation hash, and GPU UUID/runtime library identities were stable.

| Input tokens | Prompt SHA256 |
| ---: | --- |
| 128 | `8c231df4f902351f7e373bce8e5075237ecb841ef908db530d665822805e45af` |
| 512 | `34ae55544e925e7a14c1efd468127b35568403c23fec492eb82d90815231e085` |
| 2048 | `6bca257e20896b3c714614824cd513cc9d143d7217e912bb44885140a2db40af` |
| 4096 | `b7b1613c74957aa8a8e6dd9b1101d8a21164a5eb8642fff5770af722ae86dc75` |

Forced-continuation SHA256:
`4a5d20aef18eff966d1b6c97d59352efba1d6a58a159b1e88489fa3c545973ee`.

All copied last-position logits were finite across all 5120 measured
predictions. For each input length, all ten measured requests produced one
identical predicted-token stream and one identical concatenated-logits hash.
At input length 128, the timing requests agreed with the independent
natural-greedy reference on **1280/1280 top-1 predictions**.

These are same-build repeatability checks, not a comparison against an
independent implementation or full-precision model. They do not establish
numerical correctness, quantization accuracy, task accuracy, or text quality.
No cross-runtime allclose/error metrics are reported because no second runtime
or numerical reference was tested.

## Separate CUDA Dispatch Evidence

Short, untimed captures cover initialization, a 128-token prefill, and one
decode transition, without warmup. ORT profiling recorded **42 QMoE** and
**84 GroupQueryAttention** node executions on CUDA, matching 21/42 graph nodes
times two inference passes. Observed CPU node types were shape-related
`Concat`, `Gather`, and `Unsqueeze`.

A separate Nsight Systems 2025.3.2 capture (`--trace=cuda --sample=none
--cpuctxsw=none`) recorded:

| CUDA kernel family | Launches |
| --- | ---: |
| `FusedFc1FastKernel<__nv_bfloat16, 3>` | 42 |
| `Fc2RouteFastKernel<__nv_bfloat16, 3>` | 42 |
| `FinalizeRoutesFastKernel<__nv_bfloat16>` | 42 |
| `PackBitplanes` | 0 |

These are aggregate counts over the short process capture, not phase-exclusive
counts or kernel timings from the performance matrix. They confirm observed
BF16 INT3 fast-path dispatch for the short workload only; they do not prove
every longer-context request used identical kernel choices or attribute
full-model latency to one kernel. Profiling overhead is excluded from all
performance tables.

## Decode Profiling: Input Lengths 128 and 512

Completed on 2026-10-29 to investigate the approximately 60 tok/s decode
result. **The INT3 QMoE fast path is active, but its core kernels account
for only about 20% of summed GPU kernel time.** Dense BF16 matrix operations
and FP32 GEMV together account for about 51-52%. There are also many small
normalization/cast kernels and substantial launch/submission work. This is
not evidence that a scalar INT3 fallback caused the observed throughput.

### Protocol and Unprofiled Baseline

These are new diagnostic runs, not additional samples pooled into the
original performance matrix. They use physical **GPU 1**, UUID
`GPU-1d6c259c-3f83-8046-c0b6-cbeb9e062c8d`, another A100-SXM4-80GB, in the
replacement isolated `jiafa-dev` container. The ORT revision, installed
package, and all three mapped ORT library hashes match the identity tables
above. The same model files and forced continuation are used, with full
unchunked prefill, the default `kNextPowerOfTwo` arena strategy, batch size 1,
8/1 threads, and CUDA graphs disabled. Clocks and power are not locked.
The synthetic-input/routing and accuracy limitations above still apply.

For each input length:

- A fresh baseline process runs 3 warmups and 5 measured requests: **635
  measured decode transitions**, with lightweight Python step timers but
  neither Nsight nor ORT profiling enabled.
- A separate Nsight Systems **2025.3.2** process runs 3 warmups and 1 measured
  request. `cudaProfilerStart/Stop` brackets only the **127 decode
  transitions**, excluding model initialization and prefill. The capture uses
  `--trace=cuda --sample=none --cpuctxsw=none
  --capture-range=cudaProfilerApi --capture-range-end=stop`.
- A separate ORT-profile process runs 1 warmup and 1 measured request.
  Analysis selects the final 127 `model_run` windows, excluding both
  prefills and warmup decode. Its operator envelopes are CPU-side timings,
  **not GPU kernel durations**; asynchronous enqueue timings must not be
  used to rank GPU bottlenecks.

Only the unprofiled baseline supplies throughput. Nsight durations below
describe the instrumented run, not an additive decomposition of that
baseline. Profiler startup is excluded from the corrected decode timer.
At each length, the five baseline requests and the two separately profiled
requests have identical predicted-token streams and concatenated-logits
hashes; this is repeatability, not an accuracy claim.

| Input tokens | Requests | Decode transitions | Decode tok/s | Decode ms/token | Synchronized ORT run ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5 | 635 | 60.034 | 16.657 | 15.452 |
| 512 | 5 | 635 | 58.844 | 16.994 | 15.786 |

The synchronized ORT run includes runtime submission and GPU execution;
it is not GPU-only time. The remaining baseline wall time is:

| Work outside synchronized ORT run | Input 128 ms/token | Input 512 ms/token |
| --- | ---: | ---: |
| Input/output binding | 0.216 | 0.216 |
| Output retrieval and synchronous last-logits D2H | 0.299 | 0.302 |
| Finite check and CPU argmax | 0.091 | 0.092 |
| Residual: logits hashing, loop work, timer overhead, etc. | 0.599 | 0.599 |
| Total outside synchronized run | 1.205 | 1.208 |

Thus Python/output handling alone does not explain the low throughput:
even considering only the measured synchronized run gives approximately
64.7/63.3 tok/s, not a several-fold improvement. This is timer arithmetic,
not a measured optimized implementation.

### GPU Kernel-Time Breakdown

Each trace contains **194183 GPU kernel launches = 1529 per decode token**.
The following rows are disjoint groups of demangled kernel names. Percentages
use **summed GPU kernel durations**, not end-to-end wall time or hardware
utilization. Rounding can affect column sums.

| Kernel family | Launches/token | Input 128 ms/token | GPU time % | Input 512 ms/token | GPU time % |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 GEMM and split-K reduction | 462 | 5.585 | 40.05 | 5.694 | 39.45 |
| INT3 QMoE FC1 + FC2 + route finalization | 63 | 2.820 | 20.22 | 2.835 | 19.64 |
| FP32 GEMV | 1 | 1.626 | 11.66 | 1.626 | 11.26 |
| Layer normalization | 253 | 1.330 | 9.54 | 1.344 | 9.31 |
| Attention, RoPE, and KV-cache preparation | 224 | 1.136 | 8.14 | 1.481 | 10.26 |
| Cast | 336 | 0.767 | 5.50 | 0.770 | 5.33 |
| Top-k routing (`SoftmaxTopKMergeKernel`) | 21 | 0.187 | 1.34 | 0.187 | 1.30 |
| Other: add, concat, activation, multiply, gather | 169 | 0.493 | 3.54 | 0.498 | 3.45 |
| **Total kernels** | **1529** | **13.945** | **100.00** | **14.435** | **100.00** |

The BF16 group includes `ampere_bf16*`, the BF16 CUTLASS GEMM, and cuBLAS
split-K reduction kernels. The attention group includes attention execution,
`CompactKvCacheRows`, `UnpackRoPEAppend`, and `GetSequenceLengths`. The
separate routing row is not included in the QMoE-core row.

GPU-active interval union, including kernels, copies, and memsets, is
**14.185/14.683 ms per token**. First-to-last GPU activity spans
**18.509/18.386 ms per token** in the respective traces, giving active-time
fractions of **76.6%/79.9%**. These are timeline coverage fractions, **not SM
utilization**, and gaps are not automatically recoverable performance.
Profiling increases submission overhead; do not subtract these values from
the unprofiled baseline to estimate CPU overhead.

For input 128, the traced CUDA launch APIs execute 1340 `cudaLaunchKernel`,
168 `cudaLaunchKernelExC`, and 21 `cuLaunchKernel` calls per token. Their
summed host durations are approximately **7.28 ms/token**. This overlaps GPU
execution and includes tracing overhead; it is **not another 7.27 ms to add
to the kernel table**. There are also 126 GPU memsets and one
**804352-byte** last-logits D2H copy per token. Small-kernel count and
submission work therefore deserve investigation alongside arithmetic.

### What This Says About INT3 and the Model

**INT3 is not falling back.** Each trace has exactly **2667 launches each**
of `FusedFc1FastKernel<__nv_bfloat16, 3>`,
`Fc2RouteFastKernel<__nv_bfloat16, 3>`, and
`FinalizeRoutesFastKernel<__nv_bfloat16>`: 21 QMoE nodes times 127 decode
transitions. No scalar-fallback or `PackBitplanes` kernels occur in either
decode capture. FC1/FC2/finalization respectively cost
**1.739/1.005/0.076 ms/token** at input 128 and
**1.747/1.007/0.081 ms/token** at input 512.

However, **fast-path dispatch does not mean the kernel is fully optimized**.
At the pinned ORT revision, the
[INT3 fast-kernel implementation](https://github.com/microsoft/onnxruntime/blob/55ab2dfff389902bb98e3404276dc44e2c71b2ee/onnxruntime/contrib_ops/cuda/moe/qmoe_kernels.cu#L2463)
unpacks codes, dequantizes/rounds weights to BF16, and accumulates FP32 dot
products using SIMT arithmetic. It uses shared inputs, vectorized loads,
warp reductions, and fused gate/up activation; it is not an INT3 Tensor
Core GEMM. Further optimization is plausible, but its achievable speedup
has not been measured. Optimizing only this approximately 20% component
would leave most observed GPU kernel work untouched.

**The model is not uniformly INT3.** In particular, ORT profiling confirms
`/lm_head/MatMul` has FP32 inputs `[1,1,3584]` and `[3584,201088]`, once per
decode token. The weight alone is **2882797568 bytes = 2.685 GiB**. The single
FP32 GEMV per token is consistent with that output projection, but the
separate ORT and Nsight captures do not provide a direct node-to-kernel
correlation. Its approximately 1.626 ms is GPU compute, distinct from the
CPU logits-copy/argmax cost. Together with the dense BF16 kernels, this
explains why three-bit expert weights alone do not guarantee high decode
throughput.

**The input-length difference is consistent with attention/cache work.**
Baseline latency increases by 0.337 ms/token from input 128 to 512.
Attention-family GPU time increases from 1.136 to 1.481 ms/token in the
traces, including `CompactKvCacheRows` increasing from 0.152 to
0.289 ms/token. QMoE-core time is nearly unchanged. This is evidence for
context-sensitive attention/cache costs, not proof of an exact causal
decomposition: baseline and traces are separate runs with unlocked clocks.

### Optimization Experiments to Prioritize

No runtime or model optimization is included in these profiling results.
Recommended follow-ups are:

1. **Dense decode matrix operations and the FP32 vocabulary projection:**
   inspect batch-1 GEMV/GEMM dispatch and split-K choices. Evaluate reduced
   head precision/quantization only as a separate model change with numerical
   and task-accuracy validation, not as a drop-in equivalent.
2. **Graph fragmentation:** investigate the 253 normalization and 336 cast
   kernels per token, dtype propagation, and supported fusion opportunities.
   The separate ORT trace records 295 MatMul nodes per token; this differs
   from GPU launch counts because a matrix operation can launch multiple
   kernels. CPU shape operations remain present.
3. **Submission and output handling:** test binding reuse and GPU-side token
   selection. Evaluate CUDA-graph feasibility rather than assuming it:
   this model has CPU shape operations and evolving cache/sequence state.
   Preserve the full-logits checks separately when comparing a production
   token-selection path against this validation-heavy benchmark.
4. **INT3 FC1/FC2:** benchmark isolated kernels before changing unpacking,
   dequantization, tiling, or reduction strategy; preserve BF16 rounding and
   fused SwiGLU semantics and validate end-to-end accuracy.

Nsight Compute **2025.3.1** was attempted for FC1/FC2 hardware counters but
failed with **`ERR_NVGPUCTRPERM`**. No shared GPU driver permission settings
were changed. Consequently this report does **not** establish occupancy,
achieved bandwidth, instruction bottlenecks, or whether any kernel saturates
the GPU. Kernel names and elapsed times alone cannot prove those claims.

Raw baseline JSON, corrected decode-only Nsight captures/SQLite/CSV,
separate ORT traces, counter-permission failure log, and `analysis.json`
are retained locally under `private-model-decode-profile-20261029`. The analysis uses
`decode-timed-input*.sqlite`; earlier captures are retained but are not the
source of the tables above. The local harness and analysis scripts are not
added to the PR, and the raw traces contain no public download link.

## Dense MatMul Optimization Experiment (2026-10-31)

**Experimental, opt-in, and not accuracy-qualified.** This follow-up changes
only dense BF16 matrix execution, not INT3 expert packing/scales, the FP32
head, attention semantics, or model precision. It is not a production
recommendation: observed numerical differences are documented below.

### Implementation and Controls

The retained ORT decode trace identifies seven dense BF16 shapes:

| K | N | MatMul calls per decode token |
| ---: | ---: | ---: |
| 3584 | 1024 | 84 |
| 3584 | 3584 | 63 |
| 3584 | 6144 | 42 |
| 6144 | 3584 | 42 |
| 3584 | 512 | 21 |
| 3584 | 1536 | 21 |
| 1536 | 3584 | 21 |

First, the existing small-N GEMV/autotuner was tested on the original
layout. Merely enabling it did not give a stable full-model improvement;
forcing the small-N kernel was slower for some isolated shapes. It was
therefore not adopted as the optimization.

The second experiment consists of:

1. A separate model with **294 BF16 constant weights** transposed bitwise
   from `[K,N]` to `[N,K]`, and their MatMul nodes expressed as
   ORT contrib `FusedMatMul(transB=1)`. Every stored transpose was reversed
   and SHA256-checked against the source bits. This adds a **6473908224-byte**
   experiment data file on disk; original files are unchanged. No precision
   reduction or requantization is performed.
2. A new CUDA BF16 GEMV candidate: one warp computes each contiguous weight
   row, with shared activation reuse, vectorized weight loads, FP32 FMA
   accumulation and warp reduction, then BF16 output rounding.
3. Selection through the existing `ep.cuda.enable_gemm_auto_tune=1` option.
   The candidate is limited to SM 8.0+, `M=1`, `transB=1`, `alpha=1`,
   no batch transposes, `N` and `K` in `[128,8192]`, `K` divisible by 8,
   and supported alignment/layout. The existing tuner requires a measured
   5% candidate advantage over cuBLAS. Ineligible cases and transposed
   prefill (`M>1`) retain cuBLAS. Default tuning remains disabled.

The runtime change is a local patch atop ORT
`55ab2dfff389902bb98e3404276dc44e2c71b2ee`, not an upstream release or merged
PR. The candidate package still reports 1.32.0; that version string does
not identify the change. Its CUDA provider SHA256 is
`da5258e42acb1aa1e1ce14698ce04906c3c861a83c6edd0a948a26ad2b01d7fe`.
The shared-provider and pybind hashes match the original identity table.
The installed baseline package was not overwritten.

The new direct-kernel, operator-dispatch/fallback, and existing autotuner
checks passed: **24 targeted CUDA internal tests**. Isolated shape tests
also compared BF16 outputs with FP64 references. Their GPU event timings
used graph replay with an 80 MiB cache-conditioning memset before each
timed region; those microbenchmarks are not full-model throughput results.

The supplied source graph itself fails the standard ONNX checker because
`SimplifiedLayerNormalization` is absent from the declared standard-domain
opset-21 schema. The experiment graph has the same checker failure, not a
successful standard-ONNX validation. Both load and run in this ORT build;
the transpose-bit checks and actual ORT execution validate the experiment
within that runtime, not portability to another ONNX implementation.

### Unprofiled End-to-End A/B

All three conditions use the **same candidate provider build** and physical
GPU 1 from the profiling section. Clocks/power remain unlocked. Prompts,
forced continuation, 128 predictions/127 decode transitions, full unchunked
prefill, thread settings, KV reset, logits validation/hashing and timers
are otherwise unchanged.

Two rounds reverse condition and input-length order. Each fresh process
runs 3 warmups and 5 measured requests: **12 processes, 60 measured requests,
7620 decode transitions**, and 36 excluded warmup requests. Each row pools
10 requests. Nsight, isolated microbenchmarks, and separate logits probes
are excluded. All timing processes have identical mapped libraries and
GPU UUID.

| Input | Condition | Decode tok/s | Gain vs original layout | ORT prefill mean ms |
| ---: | --- | ---: | ---: | ---: |
| 128 | Original layout, cuBLAS | 59.998 | baseline | 272.175 |
| 128 | Transposed layout, cuBLAS | 60.867 | +1.45% | 272.340 |
| 128 | Transposed layout, GEMV autotune | 62.680 | +4.47% | 272.572 |
| 512 | Original layout, cuBLAS | 58.592 | baseline | 1058.862 |
| 512 | Transposed layout, cuBLAS | 59.720 | +1.92% | 1059.244 |
| 512 | Transposed layout, GEMV autotune | 60.500 | +3.26% | 1059.286 |

The combined condition is faster than baseline in both rounds at both
lengths, but two rounds are not confidence intervals or production
variability coverage. Mean prefill time increases approximately
**0.15%/0.04%**; no substantial prefill slowdown was observed in these
samples. This does not establish unchanged performance at other lengths.

A separate 128-input decode-only Nsight capture confirms **63 new GEMV
launches per token**. Total GPU kernel launches fall from 1529 in the
earlier original trace to **1361 per token** in this candidate trace.
These are instrumented observations from separate runs, not an additive
latency decomposition or evidence that every process chooses identical
tactics.

### Numerical Differences and Adoption Gate

Separate one-request probes save all 128 last-position logits vectors at
each length, using the same forced inputs. Probe timings are not used in
the throughput table. Against the original-layout condition:

| Input | Condition | Matching top-1 predictions | Logits RMSE | Max absolute logits difference | Relative L2 difference |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | Transposed layout, cuBLAS | 128/128 | 0.197533 | 2.728532 | 2.67% |
| 128 | Transposed layout, GEMV autotune | 128/128 | 0.184135 | 2.330406 | 2.49% |
| 512 | Transposed layout, cuBLAS | 127/128 | 0.227795 | 2.274748 | 3.39% |
| 512 | Transposed layout, GEMV autotune | 127/128 | 0.225908 | 2.487927 | 3.37% |

Both 512-input conditions differ at prediction index **1** (the first
decode transition); their prefill top-1 agrees. All saved logits are finite.
Even the transposed-cuBLAS condition differs numerically: bitwise-preserved
weights and mathematically equivalent MatMul do not guarantee identical
floating-point results after layout/kernel selection changes.

Within each timing process, all five requests have identical token streams
and logits hashes. Across rounds, the autotuned condition has **two distinct
logits hashes**, although its predicted-token stream is unchanged at each
length. Per-process tactic selection is not a bitwise-reproducibility
guarantee across fresh processes.

**These synthetic forced-continuation probes do not establish generation
quality or task accuracy.** Free-running generation and task evaluation
with the source tokenizer/checkpoint are required before adopting the
layout change or enabling the new path for this model. The measured
3-4% throughput gain alone is insufficient for production approval.

Local scripts, `runtime.patch`, the candidate Python package, all
per-request JSON, numerical probes, Nsight capture/SQLite/CSV, and
`summary.json` are retained under `private-model-dense-20261031`. The separate model
is `dense-transposed-20261031.onnx` beside the original model data. No
model weights or raw captures are included in this report.

## INT3 QMoE FC1/FC2 Optimization Experiment (2026-10-31)

This separate follow-up optimizes the existing **BF16 + INT3 fast path** in
ORT's CUDA QMoE kernels. It uses the **original, unchanged Private Model graph and
weights**, with dense GEMM autotuning disabled. It does not require an
Olive/export change and does not combine the dense experiment above.

### Implementation and Numerical Validation

The original dot helper multiplies each unpacked code by its scale in FP32,
rounds each weight to BF16, and converts it back to FP32 before accumulation.
The candidate instead retains the scale as BF16 and uses native **BF16x2
multiplication** to dequantize two weights together. INT3 codes are exactly
representable, allowing the same per-weight BF16 rounding. The eight-product
FP32 dot expression and its accumulation order are unchanged.

All six fast FC1/FC2 helper call sites use the typed-scale helper. Other
bit widths and dtypes retain their previous float-scale computation.
Unpacking, weight layout, routing, reduction, fused SwiGLU, output rounding,
and scalar fallback are unchanged. This remains a **SIMT kernel, not a
Tensor Core implementation**.

Validation completed:

- CUDA internal test covers all **65536 BF16 scale bit patterns**, all
  **64 INT3 code pairs**, and two activation patterns: **8388608 cases**.
  Individual dequantized weights and complete dots are compared bitwise
  for non-NaN results; NaN classification, not payload, is checked.
- Six existing BF16 SwiGLU and INT4 prepacking regression tests pass.
- The first real Private Model QMoE operator produces bitwise-identical BF16
  outputs at 1, 8, and 128 input rows.
- Separate full-model probes at both input lengths save all
  **128 x 201088 FP32 last-position logits**. Baseline and optimized arrays
  are finite and **bitwise identical**, with 128/128 matching top-1
  predictions. All timed requests also share the same logits hash and
  predicted-token stream across conditions and rounds at each length.

These checks establish equivalence for the exercised implementation and
workloads; they are not a separate task-quality evaluation or validation
on other GPU architectures.

### Unprofiled Original-Model A/B

Same A100 GPU1 and benchmark timing boundaries as above. Two rounds reverse
condition and length order, with three warmups and five measured requests
per fresh process: **40 measured requests, 5080 decode transitions**.
Each request has one prefill and 127 decode transitions, using the same
forced reference continuation. Four numerical probes and four decode-only
Nsight captures are excluded from these throughput measurements.

| Input | Baseline decode tok/s | Optimized decode tok/s | Decode throughput gain | Baseline prefill ms | Optimized prefill ms | Prefill throughput gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 59.8362 | 60.7666 | +1.56% | 272.699 | 247.935 | +9.99% |
| 512 | 58.6306 | 59.0879 | +0.78% | 1059.105 | 964.137 | +9.85% |

Decode throughput is pooled as total transitions divided by total decode
wall time; prefill latency is the mean. The prefill latency reductions are
**9.08%/8.97%**, distinct from the throughput gains in the table.

Round 1/2 decode rates are baseline **60.2083/59.4687** versus optimized
**60.7618/60.7714** at input 128, and baseline **58.5542/58.7072** versus
optimized **59.0543/59.1215** at input 512. Both rounds favor the candidate,
but the small decode gain must be interpreted with unlocked clocks and
baseline run-to-run variation; no statistical confidence interval is
claimed.

### Decode Kernel Evidence and Limits

The final candidate's separate whole-model captures give:

| Input | Kernel | Baseline GPU ms/token | Optimized GPU ms/token |
| ---: | --- | ---: | ---: |
| 128 | FC1 | 1.7371 | 1.5744 |
| 128 | FC2 | 1.0017 | 0.9426 |
| 512 | FC1 | 1.7431 | 1.5847 |
| 512 | FC2 | 1.0037 | 0.9492 |

FC1 GPU duration drops approximately **9%**, FC2 **5-6%**, and their
combined duration **7.8-8.1%**. Each capture still contains 2667 launches
each of FC1, FC2, and route finalization, with **1529 total kernels/token**
and no scalar fallback or `PackBitplanes`. This optimization reduces
arithmetic/conversion work, not launch count. Profiled GPU durations are
not an additive decomposition of the separate unprofiled wall timings.
Hardware counters remain unavailable, so bandwidth/occupancy bottlenecks
are not established.

Isolated first-QMoE wall latency is **0.2457 -> 0.2197 ms** at one row,
**0.8645 -> 0.8466 ms** at eight rows, and **12.9394 -> 11.4824 ms** at
128 rows. These limited operator samples use isolated routing/cache
workloads and must not be substituted for full-model throughput.

**Conclusion:** this is a numerically validated, modest decode improvement
with a larger prefill benefit. FC1/FC2 are only part of total decode work;
dense matrix operations, normalization/casts, attention/cache work, and
submission/output handling remain. It does not turn the approximately
60 tok/s baseline into a substantially faster decoder.

The baseline package contains the earlier default-off dense experiment,
but the original model and disabled autotuning leave its experimental
path inactive. Only the CUDA provider differs in this A/B:

- Baseline CUDA SHA256:
  `da5258e42acb1aa1e1ce14698ce04906c3c861a83c6edd0a948a26ad2b01d7fe`.
- Optimized CUDA SHA256:
  `ef2d221691a5e14276eb7b8aa7a190aed2687d963e1100588a5e9ff958502e6d`.
- Shared provider and Python binding hashes match between conditions.

Separate runtime packages, test/build logs, per-request JSON, full-logit
probes, final Nsight captures/SQLite, `summary.json`, and the QMoE-only
`runtime.patch` are retained locally under `private-model-qmoe-20261031`.
The original installed ORT package and model files are unchanged.

## Latest Upstream Main Comparison (2026-11-01)

Before opening a PR for the local QMoE experiment, upstream was refreshed
to **`a727d71c6120890b752daf5915749b1f18e180ed`**, whose tip is
[ORT PR #33689](https://github.com/microsoft/onnxruntime/pull/33689).
This version already introduces vectorized INT3 BF16x2 dequantization and
INT3 Tensor Core kernels. The local paired-dot patch is **not applied** in
this comparison; neither is the dense-layout experiment.

### Controls and Validation

A clean separate checkout was built with CUDA 13, SM80, Release, and the
previous build's runtime options. The original installed runtime, measured
candidate packages, model graph, and external weights were preserved.
The baseline is the same preserved old package used in the local QMoE
experiment, with its dense experimental path inactive.

Both conditions use the original Private Model model, GPU1, batch 1, the same prompt
and forced reference continuation, disabled dense GEMM autotuning, and
the same prefill/decode timing boundaries. Two reversed-order rounds were
followed by one adjacent A/B round because the old 128-input baseline
showed appreciable variation. Each process has three warmups and five
measured requests: **60 measured requests, 7620 decode transitions**.
All three rounds are retained; no slow baseline samples were discarded.
Four separate full-logit probes and four decode-only Nsight captures are
excluded from timing aggregates.

Eight targeted upstream INT3 tests passed without skips, covering fast-path
parity, per-weight rounding/cancellation, prefill parity, Tensor Core
session options/parity, extreme scales, tiny activations, and zero weights.
At both input lengths, the saved **128 x 201088 FP32 logits arrays are
finite and bitwise identical** between conditions: 128/128 top-1 matches,
zero maximum difference, zero RMSE, and zero relative L2 difference.
Timed-request logits hashes also match across conditions and rounds.
These remain synthetic equivalence checks, not task-quality evaluation
or validation of every shape/architecture.

### Full-Model Results

| Input | Old baseline decode tok/s | Latest main decode tok/s | Pooled decode gain | Old baseline prefill ms | Latest main prefill ms | Prefill latency reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 57.5159 | 66.3500 | +15.36% | 271.996 | 120.394 | 55.74% |
| 512 | 58.3588 | 64.6325 | +10.75% | 1059.305 | 459.665 | 56.61% |

Decode throughput is total transitions divided by total decode wall time;
prefill latency is the mean. Prefill throughput increases
**2.2592x/2.3045x**, distinct from the latency reductions.

| Input | Round | Old baseline decode tok/s | Latest main decode tok/s |
| ---: | ---: | ---: | ---: |
| 128 | 1 | 60.3089 | 66.9193 |
| 128 | 2 | 55.9478 | 66.8491 |
| 128 | 3, supplemental | 56.4831 | 65.3068 |
| 512 | 1 | 57.9118 | 64.7251 |
| 512 | 2 | 58.9557 | 64.7948 |
| 512 | 3, supplemental | 58.2188 | 64.3791 |

All rounds favor latest main. However, **the 128-input pooled gain is
influenced by the slower old-baseline rounds**; it is not a guaranteed
15.36% improvement. Round 1 alone gives approximately 10.96%. Clocks remain
unlocked, and this experiment does not isolate the cause of the baseline
variation or provide a statistical confidence interval.

### Actual Private Model Dispatch

**Private Model does not use the new INT3 Tensor Core path in these captures.**
Its quantization group size is **32**; this upstream INT3 Tensor Core
implementation requires **64**. The measured paths are instead:

- Baseline: `FusedFc1FastKernel<BF16,3>` and
  `Fc2RouteFastKernel<BF16,3>`.
- Latest main: `FusedFc1FastVecKernel<BF16,3,32>` and
  `Fc2RouteFastVecKernel<BF16,3,32>`, with packed BF16x2 dequantization.
- Route finalization changes from `FinalizeRoutesFastKernel` to
  `FinalizeRoutesVec8Kernel`. Each FC1, FC2, and finalization kernel
  still launches 21 times per decode token.

| Input | Kernel | Old baseline GPU ms/token | Latest main GPU ms/token |
| ---: | --- | ---: | ---: |
| 128 | FC1 | 1.7336 | 0.7232 |
| 128 | FC2 | 0.9974 | 0.5057 |
| 512 | FC1 | 1.7394 | 0.7292 |
| 512 | FC2 | 1.0017 | 0.5102 |

Combined FC1/FC2 GPU duration falls approximately **55%**. Summed GPU
kernel time changes from **13.8992 -> 12.2763 ms/token** at input 128 and
**14.3501 -> 12.7508 ms/token** at input 512.

Total launches fall from **1529 to 1508/token**. The traces also replace
21 `QuickGelu` and 21 elementwise multiply launches with 21 `SwiGLUKernel`
launches; cast counts remain 336/token. Routing changes from
`SoftmaxTopKMergeKernel` to `FusedInt3SoftmaxTopKGroupsKernel`.
These observations demonstrate that the comparison includes **multiple
upstream changes**, not just the local QMoE dequantization idea. It does
not attribute the full wall-time gain to one kernel or PR. Hardware
counters remain unavailable.

### Consequence for the Proposed Optimization PR

The earlier local paired-dot patch remains applicable in principle to
the older non-vectorized fast path, but that is **not Private Model's selected
path on this latest main**. Its previously reported +1.56%/+0.78% decode
gains must not be presented as gains over latest main. No redundant
optimization PR was opened; the recommended next baseline is this
unmodified upstream runtime.

Local source is clean at the revision above. Latest-main CUDA provider
SHA256 is
`0b03955cbec4fee6303cf3d93bf0db98d4bc795e63abad5d9b296e8c72eb27e5`.
The shared-provider hash matches the baseline; the Python binding changes
with upstream source. A separate package, `runtime-manifest.json`,
build/test logs, all three timing rounds, full-logit probes, Nsight
captures/SQLite, and `summary.json` are retained under
`private-model-main-20261101`.

## Refreshed Upstream Decode Profiling (2026-11-01)

Fresh captures were taken after the upstream comparison, using its
unchanged **`a727d71c6120890b752daf5915749b1f18e180ed`** runtime and original
Private Model model on A100 GPU1. At refresh time, `origin/main` had advanced to
`7f9a5e846af20a4dc36b9bd6cfc17135dd9647c5`; that intervening commit changes
only WebGPU files. We retained the explicitly pinned, already validated
CUDA build rather than relabeling it as a newly built revision.

### Capture Method and Unprofiled Timing

At each input length:

- A fresh **unprofiled** process runs three warmups and five requests,
  recording decode step timing.
- A separate Nsight Systems process runs three warmups and one request,
  with capture restricted to the **127 decode transitions** through
  `cudaProfilerStart/Stop`; initialization and prefill are excluded.
- A separate ORT profiling process runs one warmup and one request.
  Node accounting selects only the final 127 decode `model_run` events.

Each request predicts 128 tokens: one prefill prediction and 127 decode
transitions. Inputs, forced continuation, original-model graph hash, GPU
UUID, and all mapped ORT binary hashes agree across the three process
types. Logits hashes match the preceding upstream comparison at each
length. Dense GEMM autotuning and local experimental patches remain off.
NVML checks observed GPU process identity throughout; clocks are unlocked.

| Input | Fresh unprofiled decode tok/s | Decode wall ms/token | Synchronized inference ms/token | Outside synchronized inference ms/token |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 66.7912 | 14.972 | 13.758 | 1.214 |
| 512 | 63.2507 | 15.810 | 14.431 | 1.379 |

These are **new single-process timing samples**, not replacements for the
three-round A/B aggregates above. The 512 rate is lower than its previous
64.6325 tok/s aggregate; run-to-run variation remains observable.

Outside synchronized inference, binding takes **0.216/0.248 ms/token**,
output retrieval/copy **0.306/0.376**, finite-logits checking and CPU argmax
**0.091/0.119**, and other Python bookkeeping **0.600/0.635**, respectively.
The synchronized-inference measurement includes host submission and waits;
it is not GPU kernel time alone. These diagnostic boundaries describe the
current full-logits validation harness, not an optimized production
token-selection loop.

### Refreshed GPU Kernel-Time Breakdown

Each decode trace contains **191516 launches = 1508 kernels/token**.
Rows are disjoint groups of demangled names; percentages use **summed
GPU kernel durations**, not wall time or hardware utilization.

| Kernel family | Launches/token | Input 128 ms/token | GPU time % | Input 512 ms/token | GPU time % |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 GEMM and split-K reduction | 462 | 5.571 | 45.31 | 5.695 | 44.69 |
| FP32 GEMV | 1 | 1.630 | 13.26 | 1.627 | 12.76 |
| Layer normalization | 253 | 1.334 | 10.84 | 1.327 | 10.41 |
| Attention, RoPE, and KV-cache preparation | 224 | 1.129 | 9.19 | 1.451 | 11.39 |
| Cast | 336 | 0.750 | 6.10 | 0.757 | 5.94 |
| INT3 QMoE FC1 + FC2 + route finalization | 63 | 1.308 | 10.63 | 1.311 | 10.29 |
| Top-k routing | 21 | 0.142 | 1.15 | 0.150 | 1.17 |
| Other: add, concat, activation, multiply, gather | 148 | 0.433 | 3.52 | 0.426 | 3.35 |
| **Total kernels** | **1508** | **12.296** | **100.00** | **12.744** | **100.00** |

FC1/FC2/finalization individually cost **0.724/0.506/0.078 ms/token** at
input 128 and **0.728/0.508/0.076** at input 512. Each kernel launches
2667 times per trace. Names confirm `FusedFc1FastVecKernel<BF16,3,32>`,
`Fc2RouteFastVecKernel<BF16,3,32>`, and `FinalizeRoutesVec8Kernel`;
no scalar fallback or `PackBitplanes` occurs. Group size 32 keeps Private Model
on the **vectorized SIMT path**, not the upstream group-64 INT3 Tensor
Core path.

Compared with the earlier 2026-10-29 captures, QMoE core falls from
**2.820/2.835 to 1.308/1.311 ms/token**, and its share from approximately
20% to approximately 10%. Dense BF16 remains approximately 5.6 ms/token;
its share rises because other work became faster. These are separate
captures with unlocked clocks, not an exact additive wall-time attribution.

### Submission, Output Head, and ORT Evidence

The GPU-active union of kernel/copy/memset intervals is
**12.538/12.963 ms/token**. First-to-last GPU activity spans
**18.914/23.578 ms/token** in the instrumented captures, giving timeline
coverage of **66.3%/55.0%**. These are not SM-utilization measurements or
directly recoverable idle-time budgets. Nsight decode wall time is
**18.923/23.588 ms/token**, appreciably above the unprofiled samples.
Do not subtract profiled GPU spans from unprofiled wall times to infer
CPU overhead.

CUDA launch APIs execute **1319 `cudaLaunchKernel`, 168
`cudaLaunchKernelExC`, and 21 `cuLaunchKernel` calls/token**. Their summed
host durations are approximately **7.69/9.10 ms/token**, overlapping GPU
execution and including tracing overhead; they must not be added to the
GPU table. Each token also has 126 GPU memsets, two tiny H2D copies totaling
12 bytes, and one **804352-byte** last-logits D2H copy.

Separate ORT traces confirm **295 MatMul, 336 Cast, 253
SimplifiedLayerNormalization, 42 GroupQueryAttention, and 21 QMoE
node calls/token**. Every QMoE and GroupQueryAttention event uses CUDA.
CPU-provider nodes still include `Concat`, `Gather`, and `Unsqueeze`.
ORT node durations represent instrumented host/operator work and are
**not GPU kernel durations**; for example, the asynchronously submitted
head can have a short ORT event despite its much longer GPU GEMV.

ORT again records `/lm_head/MatMul` with FP32 inputs `[1,1,3584]` and
`[3584,201088]`, once per decode token. Its **2.685-GiB FP32 weight** is
consistent with the single approximately 1.63-ms FP32 GEMV. Separate ORT
and Nsight captures still do not provide a direct node-to-kernel correlation.

### Remaining Priorities

1. **Dense matrix operations and the vocabulary projection:** dense BF16
   plus FP32 GEMV account for approximately **57-59%** of summed GPU
   kernel time. Revisit dispatch/layout tactics on this new baseline.
   The older dense experiment's gains are not established as additive
   over this build, and its numerical differences still block adoption.
   Head precision/quantization would be a separate model/accuracy change.
2. **Normalization and cast fusion:** 589 launches/token and approximately
   **16-17%** of GPU kernel time remain. Preserve existing dtype conversions,
   FP32 normalization arithmetic, and output rounding when testing fusion.
3. **Submission and output handling:** evaluate binding reuse and GPU-side
   token selection against the measured 1.2-1.4 ms outside synchronized
   inference. Keep full-logits numerical probes separate. CUDA-graph
   feasibility requires validation of CPU shape nodes and evolving cache
   state; it is not established by launch count alone.
4. **Attention/cache work and further QMoE changes:** attention scales with
   context length, while QMoE core is nearly unchanged between these short
   contexts. New QMoE work should target the selected vectorized path or
   explicitly validate group-32 Tensor Core support, not resubmit the old
   paired-dot patch for a path Private Model no longer selects.

Nsight Compute was retried on the current FC1/FC2 vectorized kernels using
the extracted real QMoE operator, but again failed with
**`ERR_NVGPUCTRPERM`**. Driver permissions were not changed. The
counter-attempt operator timing is instrumented and **not a performance
result**. Occupancy, achieved bandwidth, and instruction bottlenecks
therefore remain unproven.

Fresh timing JSON, decode-only Nsight captures/SQLite/CSV, separate ORT
profiles, counter-denial log, copied runtime manifest, and `analysis.json`
are retained locally under `private-model-decode-profile-20261101`. Analysis checks
runtime/input/logits identities, QMoE dispatch counts, and node accounting;
the extended analyzer also reproduces both historical 2026-10-29 results.
Only this report is submitted; raw captures, weights, and local scripts
are not included in the PR.

## Equivalent Norm/Cast Fusion Experiment (2026-11-01)

This local experiment follows the refreshed profiling above. It removes
standalone dtype-conversion launches around normalization while retaining
the existing **FP32 normalization arithmetic and BF16 output rounding**.
It does not replace RMSNorm with a lower-precision approximation.

### Implementation, Model Conditions, and Isolation

The baseline is the preserved `a727d71c6120890b752daf5915749b1f18e180ed`
runtime above, not a fresh claim about the latest upstream revision. The
candidate is built in a separate worktree with matched Release/CUDA 13/SM80
options. Its CUDA source changes only register and instantiate the existing
normalization implementation for **BF16 input / FP32 scale and output** and
**FP32 input / BF16 scale and output**. Reduction, epsilon, multiplication
order, and final conversion are unchanged. The deprecated LayerNormalization
opsets 1-16 share these registrations; this does not add mixed types to the
standard opset-17 LayerNormalization schema.

Separate exported graphs test two scopes:

| Condition | Rewritten norms | Removed Cast nodes | Losslessly converted scale tensors |
| --- | ---: | ---: | ---: |
| Q/K only | 84 | 168 | 84 |
| All eligible norms | 252 | 336 | 168 |

Q/K fusion removes both surrounding casts using the already available
BF16-input/BF16-output kernel. The full candidate also removes the leading
BF16-to-FP32 cast at 84 post-norms and the trailing FP32-to-BF16 cast at
84 pre-norms, using the two new mixed-type registrations. FP32 residual
adds, the final FP32 normalization, and the FP32 vocabulary head remain
unchanged.

Trailing-cast removal requires the scale values to be **exactly
BF16-representable**. All rewritten Private Model FP32 scales meet this condition;
the exporter verifies the low 16 bits are zero and rejects non-finite or
non-representable values instead of rounding them. This is a model-specific
lossless conversion, not permission to downcast arbitrary FP32 norm weights.
The original graph, external weights, installed runtime, and preserved
baseline package are not overwritten.

Both baseline and candidate explicitly disable
`GroupQueryAttentionPreNormFusion` through
`optimization.disable_specified_optimizers`. Otherwise rewriting Q/K to
BF16 could enable a different attention-normalization fusion and confound
this equivalence experiment. Original-model logits with this gate match
the preceding upstream probes bitwise. Dense autotuning remains off.
In this initial experiment, no automatic ORT graph rewrite was implemented; the graph
changes are explicit, separately exported experimental models.

| Artifact | SHA256 |
| --- | --- |
| Q/K-only graph | `86383e1ad3d35ef6d8da90d2f6b4c3a7729efae95ccb8edac6274eb115c86174` |
| Full candidate graph | `153e5afe5054b455789b54a43aec88f845ba656064546ccdc0aec785c8445353` |
| Candidate CUDA provider | `1e1eeed47ec031c0ed8b3d589758a13c198c9e8e6add022510f43d8f20f44063` |

The shared-provider and Python-extension hashes match the preserved baseline.
The exported graphs have the same experimental
SimplifiedLayerNormalization checker limitation as the source graph;
**ONNX checker success is not claimed**. Runtime loading and numerical
probes below validate the exercised graphs instead.

### Numerical and Existing-Behavior Checks

- **43 LayerNorm regression tests passed**, including the two new CUDA-only
  mixed-type tests, each exercising both normalization operators.
- A separate GPU-bound operator matrix passed **240 cases and
  152,265,648 output elements**: BF16/BF16, BF16/FP32, and FP32/BF16;
  both operators; ten shapes covering decode and 128/512-token prefill;
  normal, signed-zero, tiny, and mixed-magnitude inputs. Non-NaN outputs
  match bitwise and NaN classifications match. Auxiliary statistics are
  also checked except for the multi-axis simplified-norm case, where the
  legacy schema's auxiliary-shape inference is inconsistent with runtime.
- At each input length, both exported candidates produce finite
  **`[128,201088]` FP32 logits identical bitwise to the baseline**, including
  all 128 predictions. Maximum absolute difference is zero, not merely
  agreement of argmax tokens.
- An additional original-graph/new-runtime control also matches full logits
  bitwise. Two reversed-order timing rounds of that control show runtime-only
  decode changes of **-0.15% / -0.48%** at input 128/512, rather than a
  speedup from replacing the provider library alone.

These checks establish equivalence for the tested synthetic inputs and GPU,
not a task-accuracy evaluation or a guarantee for other models/architectures.

### Unprofiled Three-Condition Results

The same A100 GPU 1, UUID `GPU-1d6c259c-3f83-8046-c0b6-cbeb9e062c8d`,
prompt rule, forced continuation, cache settings, and synchronization
definitions are retained. Two rounds reverse both condition and input-length
order; each fresh process uses three warmups and five measured requests.
Each table row pools ten requests and 1270 decode transitions. The primary
matrix comprises **60 measured requests / 7620 decode transitions**; the
additional runtime-only control comprises 40 requests and is not pooled
into this table. Numerical probes and profiler captures are separate.
Clocks and power are not locked; these are observed gains, not confidence
intervals or architecture-wide guarantees.

| Input | Condition | Decode tok/s | Decode gain | Prefill ms | Prefill latency reduction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | Original baseline | 66.922 | - | 120.434 | - |
| 128 | Q/K only | 68.198 | +1.91% | 119.915 | 0.43% |
| 128 | All eligible norms | 69.462 | **+3.79%** | 119.729 | 0.59% |
| 512 | Original baseline | 64.183 | - | 459.947 | - |
| 512 | Q/K only | 66.180 | +3.11% | 458.338 | 0.35% |
| 512 | All eligible norms | 68.375 | **+6.53%** | 459.347 | 0.13% |

| Input | Condition | Round 1 decode tok/s | Round 2 decode tok/s |
| ---: | --- | ---: | ---: |
| 128 | Original baseline | 66.871 | 66.973 |
| 128 | Q/K only | 67.890 | 68.508 |
| 128 | All eligible norms | 69.494 | 69.430 |
| 512 | Original baseline | 64.412 | 63.956 |
| 512 | Q/K only | 66.145 | 66.216 |
| 512 | All eligible norms | 68.435 | 68.315 |

Both rounds improve decode. Prefill changes remain below 0.6%; a meaningful
prefill speedup is not established by this experiment.

### Actual Launch Reduction and Remaining Bottleneck

Decode-only Nsight captures contain 127 transitions each. The table reports
summed GPU kernel durations, **not unprofiled end-to-end latency**.

| Input | Condition | Cast launches/token | Norm launches/token | All kernel launches/token | Norm + cast ms/token | All kernels ms/token |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | Original baseline | 336 | 253 | 1508 | 2.098 | 12.388 |
| 128 | Q/K only | 168 | 253 | 1340 | 1.803 | 12.068 |
| 128 | All eligible norms | 0 | 253 | 1172 | 1.604 | 11.825 |
| 512 | Original baseline | 336 | 253 | 1508 | 2.086 | 12.751 |
| 512 | Q/K only | 168 | 253 | 1340 | 1.786 | 12.414 |
| 512 | All eligible norms | 0 | 253 | 1172 | 1.631 | 12.380 |

The full candidate removes **336 launches/token (22.3% of all kernel
launches)**. Normalization itself is slightly slower with the mixed/BF16
accesses, but eliminating the standalone casts reduces combined norm/cast
GPU time by about **22-24%**. The earlier 16-17% norm/cast share was not
an entirely removable latency budget.

Separate ORT profiles confirm 253 normalization, 295 MatMul, 42
GroupQueryAttention, and 21 QMoE node calls per transition in both graphs;
only the 336 Cast calls disappear. The 84 Add calls remain. Relevant
compute nodes stay on CUDA, and Nsight still selects 21 calls/token of
each QMoE fast-vector FC1, FC2, and finalize kernel. No attention-norm
fusion or Dense dispatch change is responsible for the launch reduction.

This is a useful, lower-risk decode improvement, not a solution to the
remaining Dense/head bottleneck. A production ORT change would still need
a correctly guarded graph transformer or an explicit export integration,
broader testing, and a decision about the attention-fusion interaction.
Build logs, source patch, both independent graphs and manifests, runtime
hashes, operator checks, complete logits, all timings, Nsight captures,
ORT profiles, `summary.json`, and `controls-summary.json` are retained
locally under `private-model-norm-cast-20261101`. Raw artifacts and runtime/model
changes are not added to this report-only worktree.

### Opt-In ORT Integration on Newer Main

The follow-up implementation is published for review in
[ORT Draft PR #33801](https://github.com/microsoft/onnxruntime/pull/33801).
It is based on newer main
`2927f3b3fcd2012394c8a4bb8d2d18f81cc0f768`, rather than the a727 runtime
used by the independent-export experiment above.

The integration works directly on the **unchanged original model**, without
an offline graph export. Enable it with:

```python
session_options.add_session_config_entry(
    "optimization.enable_bf16_norm_cast_fusion", "1"
)
```

It is default-off and runs at optimization level 2 or higher. The transformer
targets CUDA SimplifiedLayerNormalization with FP32 scale/output and optional
surrounding BF16 casts. Leading-cast removal retains the FP32 scale;
trailing-cast removal requires a constant, finite, exactly BF16-representable
scale and creates a separate initializer. Shared/exposed FP32 intermediates,
overridable scales, auxiliary-output norms, and non-CUDA providers are guarded.
The two mixed-type kernel registrations retain the existing arithmetic.

Enabling the option also excludes **CUDA** GroupQueryAttentionPreNormFusion
from the default optimizer pipeline, avoiding a newly eligible attention
normalization with different arithmetic. Its WebGPU path is unchanged.
The disabled/default option preserves the existing pipeline; this deliberate
opt-in interaction is documented in the public session-option definition.

The new build passed **26 selected optimizer tests** (five new tests,
including 19 structural guard cases, plus existing attention-pre-norm tests),
**43 normalization regressions**, and the same **240-case / 152,265,648-element**
standalone matrix. WebGPU numerical tests were excluded because this build
does not include that provider. In-memory external initializers have a
dedicated structural test; original-model probes exercise file-backed scales.

Full `[128,201088]` FP32 logits are finite and identical bitwise with the
option off/on at both lengths. All timed and profiled last-logits hashes
match. Default-off logits also match the preserved a727 probes bitwise.
Both original model-file hashes remain unchanged.

This fresh A/B uses the **same patched runtime and original graph** in both
conditions; only the option differs. No explicit optimizer-disable override
or Dense tuning is used. Two reversed-order rounds retain three warmups and
five measured requests per fresh process: **40 measured requests / 5080
decode transitions**, with ten requests per row. The A100 GPU, synthetic
prompts, forced continuation, and metric definitions are unchanged.

| Input | Option off decode tok/s | Option on decode tok/s | Gain | Off/on prefill ms |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 66.366 | 69.993 | **+5.47%** | 120.532 / 119.484 |
| 512 | 64.506 | 68.009 | **+5.43%** | 459.295 / 459.933 |

The per-round decode rates are **66.656/66.078 -> 70.014/69.972**
at input 128 and **64.917/64.100 -> 67.784/68.236** at input 512.
Both rounds improve decode. Prefill remains approximately unchanged, with
no consistent gain. These fresh results do not replace the earlier export
experiment or imply additive gains over its different baseline.

Separate Nsight/ORT profiles confirm **336 -> 0 Cast calls/launches** and
**1508 -> 1172 total kernel launches** per transition. Normalization remains
253 calls, MatMul 295, Add 84, GroupQueryAttention 42, and QMoE 21.
All non-norm/cast kernel names and counts match. Combined norm/cast GPU time
falls from **2.095/2.096 to 1.614/1.610 ms/token**, approximately 23%;
instrumented GPU durations are not substituted for unprofiled throughput.

Candidate CUDA-provider SHA256 is
`6fc043ef8b0b4a9fcab76e0847a4a18782e50ee462d1123e97471519bdacb04c`;
Python-extension SHA256 is
`c19e23cc54659ffb568a3d7c84b83d2508a7474de90f2406c6d4b4fccd21251d`.
The shared-provider hash is unchanged. Source patch, build/test logs,
runtime manifest, original-model logits, all timings, traces,
`summary.json`, and `verification.json` are retained locally under
`private-model-norm-cast-pr-20261101`. The source implementation is in the separate
ORT PR; raw artifacts remain private/local.

## Python-Only Q/K Norm Export on Unmodified ORT (2026-11-02)

Following review feedback, this separate experiment removes the Q/K norm
Casts with [a standalone Python script](scripts/fuse_bf16_norm_cast.py),
without the C++ transformer, new kernel registrations, or the opt-in flag.
Both conditions use the **same unmodified upstream ORT build at
`a727d71c6120890b752daf5915749b1f18e180ed`**. This is a pinned upstream
build, not a claim about every released ORT package.

### Rewrite and Reproduction

The script recognizes complete chains by topology and tensor types, not
private model/node names:

```text
BF16 -> Cast(FP32) -> SimplifiedLayerNormalization -> Cast(BF16)
```

It replaces them with a BF16-input/BF16-output norm using the existing
`LayerNorm<BFloat16, float, BFloat16, true>` CUDA implementation. Internal
accumulation and normalization arithmetic remain FP32. BF16-to-FP32 and
FP32-to-BF16 conversions still happen inside the kernel; there are no
separate Cast operators or materialized FP32 intermediate tensors for
these complete chains.

The supplied graph has **84 eligible Q/K norms**, so this export removes
**168 Casts** and creates 84 exactly representable BF16 scale initializers.
The remaining 169 norms are unchanged. In particular, this script does
**not** remove the other 168 Casts adjacent to mixed-input/output norms:
those require kernel combinations absent from this pinned upstream build.
This result is narrower than the 336-Cast integration above.

Dependencies are ONNX and NumPy (tested with 1.22.0 and 2.5.0). Run from
the recipe repository root; the model filenames below are aliases:

```bash
python scripts/fuse_bf16_norm_cast.py \
  --source /path/to/private_int3_model.onnx \
  --output /path/to/private_int3_model.qk_bf16.onnx \
  --expected-casts 168
```

The output must be a **new filename beside the original model** so that
relative external-weight references remain valid. The script writes a
separate model protobuf; it never rewrites the original external weights.
It preserves shared FP32 scales and exposed/shared intermediates, rejects
inexact or nonfinite scale conversions, and only handles norms with one
output and FP32 stash type. Unsupported chains are counted and skipped;
an unexpected removal count or no eligible chains raises an error before
writing. Graphs with subgraphs are explicitly rejected to avoid unsafe
implicit-capture rewrites.

If a scale will be replaced through session `AddInitializer`, exclude
its original name with `--exclude-initializer SCALE_NAME` (repeatable).
An offline rewrite cannot know future session overrides; overriding the
old scale name after conversion would not update the new BF16 initializer.
The expected Cast count must reflect any exclusions.

The numerical comparison explicitly disables
`GroupQueryAttentionPreNormFusion` in **both** conditions:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry(
    "optimization.disable_specified_optimizers",
    "GroupQueryAttentionPreNormFusion",
)
session = ort.InferenceSession(
    "/path/to/private_int3_model.qk_bf16.onnx",
    sess_options=options,
    providers=["CUDAExecutionProvider"],
)
```

This standard session setting requires no ORT source changes. It avoids
enabling a different attention-normalization fusion after graph rewriting;
the measured bitwise equivalence below should not be attributed to an
uncontrolled default optimizer pipeline. Do not set
`optimization.enable_bf16_norm_cast_fusion` for this Python-only experiment.

### Validation and Fresh A/B

[Twelve synthetic safety/CLI tests](scripts/test_fuse_bf16_norm_cast.py)
pass, including signed-zero scale preservation, inexact/nonfinite scales,
FP16/unknown types, shared/exposed intermediates, dynamic/overridable and
excluded scales, auxiliary outputs, stash type, subgraphs, initializer-name
collisions, external-data preservation, count checks, and overwrite safety:

```bash
python -m unittest discover -s scripts -p test_fuse_bf16_norm_cast.py -v
```

Actual ORT CUDA load/run probes at input lengths 128 and 512 produce finite,
**bitwise-identical FP32 logits of shape `[128,201088]`**: all vocabulary
entries for 128 output steps, not all prompt-position logits. Timed
last-logits hashes also match across conditions. These synthetic-prompt,
forced-continuation checks are not a task-accuracy benchmark or a proof
for every input/device. The supplied model uses an experimental norm
operator not recognized by the standard ONNX checker at its opset; this
is runtime validation, not a claim that `onnx.checker` passed.

Fresh timing on the same A100 uses two reversed-order rounds, three
warmups and five measured requests per process, input lengths 128/512,
and output length 128: **40 measured requests / 5080 decode transitions**.
No logits-dump or profiler overhead is included in these timings.

| Input | Original decode tok/s | Python-rewritten decode tok/s | Gain | Original/rewritten prefill ms |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 66.267 | 68.360 | **+3.16%** | 120.426 / 119.652 |
| 512 | 65.085 | 66.641 | **+2.39%** | 459.507 / 458.657 |

Per-round rates are **66.186/66.348 -> 68.145/68.577** at input 128
and **65.239/64.932 -> 66.279/67.008** at input 512. Decode improves
in both rounds; the small prefill differences are not evidence of a
substantial prefill improvement. These fresh measurements are not directly
comparable or additive to the earlier patched-runtime results.

Separate ORT decode profiles confirm **336 -> 168 executed Cast nodes
per transition**, while normalization remains 253, MatMul 295, Add 84,
GroupQueryAttention 42, and QMoE 21. No Nsight launch-count claim is made
for this new experiment. The original graph SHA256 and original weight
file size/mtime are unchanged.

| Artifact | SHA256 |
| --- | --- |
| Rewritten graph | `e345e3a89ecad8dea235a335d19d76518f41b4a6508648271b65fbffa2785dc9` |
| Unmodified CUDA provider | `0b03955cbec4fee6303cf3d93bf0db98d4bc795e63abad5d9b296e8c72eb27e5` |
| Unmodified Python extension | `e6f0a54229c90330aed28b280e347c9a31d3e34a0f42b4735c9b9c93c75c16e5` |

The shared-provider hash matches the earlier upstream-main build.
Complete logits, timing processes, ORT profiles, export/runtime identities,
and `summary.json` are retained locally under
`private-model-python-qk-20261102`. Raw model, weights, logits, and profiles
are not published. Only the reusable script, its tests, and this anonymized
report are included in this follow-up.

## Mixed INT4/INT8 Weight-Only Export (2026-11-03)

This local follow-up starts from the Python-only Q/K export above, rather
than the original graph. It quantizes 273 ordinary projection MatMuls to
INT4 `MatMulNBits`, 21 router MatMuls and one FP32 output-head MatMul to
INT8 `MatMulNBits`, and one FP32 embedding to INT8
`GatherBlockQuantized`. All 21 INT3 QMoE nodes and their retained source
weight tensors are unchanged. Activations and output interfaces remain
BF16/FP32; this is weight-only quantization, not integer-only inference.

The local exporter uses ORT's existing
INT4/INT8 symmetric RTN packers with block size 32. The embedding uses
per-block symmetric rounding with unsigned byte storage offset by 128.
It preserves the original graph and weights. The measured graph stores
new quantized tensors separately and references unchanged external INT3
expert tensors. The timings below precede the subsequent byte-preserving
sidecar compaction; the table in this section is not a post-compaction
measurement. The later residual-fusion experiment below measures the
compacted mixed-quantized baseline separately on a newer runtime.
Session `AddInitializer` overrides must not be reused under old weight
names after quantization.

Four local CUDA and storage tests pass, covering
INT4/INT8 MatMulNBits with FP32/BF16 inputs, decode/prefill-sized batches,
INT8 embedding with negative indices, zero weights, invalid dimensions,
and byte-preserving external-data compaction.

### Fresh Performance Comparison

Both conditions use the same unmodified upstream ORT binaries at
`a727d71c6120890b752daf5915749b1f18e180ed`, the same A100, synthetic prompts,
forced continuations, and explicit `GroupQueryAttentionPreNormFusion`
exclusion. Dense auto-tuning and the C++ norm/cast opt-in are disabled.
The baseline already includes the 168-Cast Q/K removal; these gains are
not gains relative to the original unmodified graph.

Two reversed-order rounds, three warmups and five measured requests per
fresh process give 40 measured requests and 5080 decode transitions.
Output length is 128 (127 timed decode transitions per request).
Separate numerical/profiling workers are excluded from throughput.

| Input | Q/K-export baseline tok/s | Mixed INT4/INT8 tok/s | Decode gain | Baseline prefill ms | Quantized prefill ms | Prefill latency change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 68.875 | 84.045 | **+22.03%** | 119.849 | 129.026 | **+7.66% slower** |
| 512 | 66.456 | 81.225 | **+22.22%** | 459.896 | 467.534 | **+1.66% slower** |

Per-round decode rates are 68.875/68.875 -> 84.426/83.668 at input 128
and 66.132/66.784 -> 80.666/81.791 at input 512. Both rounds improve
decode; the export is not a universal prefill speedup.

Separate ORT profiling confirms 295 MatMulNBits, one GatherBlockQuantized,
and 21 QMoE calls per decode transition, all on CUDA. Normalization remains
253 calls and Cast remains 168 calls. CPU placement is limited to observed
shape-related Gather/Unsqueeze/Concat nodes, not quantized weight operators.
This is ORT-node dispatch evidence, not a new Nsight kernel-time analysis.

### Numerical Changes and Adoption Gate

This is **lossy** quantization. Fresh full-vocabulary FP32 logits
`[128,201088]` are finite at both input lengths, but are not bitwise equal:

| Input | Relative logit L2 error | Logit RMSE | Maximum absolute error | Matching top-1 steps |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 14.38% | 1.0631 | 9.2698 | 127/128 |
| 512 | 15.80% | 1.0607 | 8.4289 | 126/128 |

The top-1 comparison is over matched synthetic/forced-continuation states,
not independent free-running generations. It is not a claim of 98–99%
task accuracy, nor evidence that routing decisions or downstream quality
are preserved. Representative task/perplexity evaluation with the correct
tokenizer remains necessary before adoption. No calibration corpus or
original source checkpoint was supplied for this RTN experiment.

Measured-process last-logits hashes are repeatable within each condition;
they intentionally differ across conditions. Complete per-process timings,
numerical probes, runtime hashes, weight-level error records, and CUDA ORT
profiles are retained locally under `private-model-mixed-quant-20261103`.
The original model, weights, quantized artifacts, and raw logits remain
private/local.

## Residual Add + RMSNorm Fusion on Latest Main (2026-11-03)

This follow-up uses a clean, separate build of unmodified upstream ORT
`a043a5fd70d01971b9953b738078e1ac6dec2153` for **both** conditions.
There are no local runtime-source changes. The baseline is the compacted
mixed INT4/INT8 graph from the previous experiment, including the earlier
Q/K Cast removal. The candidate is an offline Python graph rewrite:

- Replace 84 FP32 `Add` + `SimplifiedLayerNormalization` pairs with 84
  `com.microsoft::SkipSimplifiedLayerNormalization` nodes.
- Preserve residual sums needed elsewhere through optional output 3 of
  the fused operator; 83 of the 84 pairs need this output.
- Preserve all initializers byte-for-byte, all quantization parameters,
  295 MatMulNBits nodes, the INT8 embedding, 42 GQA nodes, and 21 INT3 QMoE
  nodes. Graph node count decreases from 1326 to 1242.
- Retain the original and baseline graphs. The candidate references the
  baseline's compacted external-data sidecar rather than duplicating weights.

Six structural rewrite tests and 16 synthetic CUDA cases pass. The CUDA
cases cover 1/3/128/512 rows at hidden size 3584 with normal, zero, tiny,
and wide-magnitude inputs. Residual sums match bitwise; normalized outputs
pass `rtol=2e-5, atol=2e-6`. These small-operator checks do not establish
full-model numerical equivalence.

### Runtime Identity and Controlled Performance

The clean Release build uses CUDA 13.0, SM80, and cuDNN 9.23.2. Both
conditions import the same staged Python package on A100-SXM4-80GB,
physical GPU 1, UUID `GPU-1d6c259c-3f83-8046-c0b6-cbeb9e062c8d`.
Mapped runtime-library hashes are verified in every worker:

| Library | SHA256 |
| --- | --- |
| `onnxruntime_pybind11_state.so` | `8cca92b96ca519dad4a80b1c0099bab27bfa60e6baaf43f66f25caf861cb0854` |
| `libonnxruntime_providers_cuda.so` | `b1b6409c4cf37c343cd2d9b3b42c95c6c753119ee3ab96bb9e008f2c04490c0e` |
| `libonnxruntime_providers_shared.so` | `17f7cf47ad0d7b5ac895ae588fd62c7f85a13842588161b6a24c7d480f062be4` |

Baseline graph SHA256:
`3da900dfd9a03a914d079ae8324a3fc86f0e8f5bc592a68023448c27a4259d84`.
Candidate graph SHA256:
`641106981bf0a8c4888c18f9aff131f4cd82cb58cd0d40b62d4004041bb78403`.

The protocol uses the same batch-1 synthetic prompts and forced continuation,
128 predictions per request, two reversed-order rounds, and three warmups
plus five measured requests per fresh process. Eight timing processes give
40 measured requests and 5080 decode transitions, excluding warmups,
numerical probes, and profiling. `GroupQueryAttentionPreNormFusion` is
explicitly disabled in both conditions; dense auto-tuning, the C++ norm/Cast
opt-in, CUDA graphs, and ORT profiling are disabled for timing.

Two preliminary attempts were rejected by the additional-GPU-process guard;
they contribute no timing results. The completed timing processes pass the
same idle-before-start and sampled compute-process isolation checks.
Sampling cannot exclude unsampled activity, and clocks are not locked.

| Input | Mixed-quantized baseline tok/s | Residual-fused tok/s | Decode gain | Baseline prefill ms | Fused prefill ms | Prefill latency change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 83.449 | 87.028 | **+4.29%** | 128.828 | 128.434 | -0.31% |
| 512 | 80.653 | 83.366 | **+3.36%** | 466.088 | 465.778 | -0.07% |

Per-round decode rates are 83.656/83.242 -> 86.862/87.194 at input 128
and 81.120/80.190 -> 83.177/83.556 at input 512. Both rounds improve
decode. Prefill differences are small and should be treated as essentially
unchanged, not a demonstrated prefill optimization. These ratios use only
the fresh latest-main baseline, not the historical 84.045/81.225 tok/s
measurement or the original model.

Separate ORT profiles confirm, per decode transition:

| CUDA operator | Baseline | Residual-fused |
| --- | ---: | ---: |
| Add | 84 | 0 |
| SimplifiedLayerNormalization | 253 | 169 |
| SkipSimplifiedLayerNormalization | 0 | 84 |
| Cast | 168 | 168 |
| MatMulNBits | 295 | 295 |
| GatherBlockQuantized | 1 | 1 |
| GroupQueryAttention | 42 | 42 |
| QMoE | 21 | 21 |

This confirms 84 fewer executed ORT CUDA nodes per transition, not a
Nsight-verified count of GPU kernel launches. This rewrite removes no
additional Casts.

### Full-Model Numerical Differences and Adoption Gate

Full-vocabulary FP32 logits `[128,201088]` are finite at both input lengths.
The new-runtime baseline matches the previous-runtime mixed-quantized
logits **bitwise**, but the residual-fused candidate does not:

| Input | Relative logit L2 error vs mixed baseline | Maximum absolute error | Matching top-1 steps |
| ---: | ---: | ---: | ---: |
| 128 | 3.6596% | 2.7367 | 128/128 |
| 512 | 3.6244% | 2.1494 | 128/128 |

The fused FP32 kernel has different reduction/arithmetic ordering.
Full-model differences can be amplified by subsequent BF16 rounding and
MoE routing; this experiment does not isolate the amplification mechanism.
**This is not the earlier lossless Q/K Cast removal.** Top-1 agreement is
only over the matched synthetic/forced-continuation states, not
free-running generations or a task-accuracy benchmark. These differences
are relative to the already lossy mixed-quantized baseline, not the original
model, and must not be added to the earlier error percentages.

Representative task/perplexity evaluation with the correct tokenizer
remains an adoption gate. The candidate does not replace the baseline.
The local graph rewrite, tests, raw per-process timings, numerical probes,
runtime manifest, and ORT profiles are retained locally under the report
alias `private-model-skipnorm-20261103`; model artifacts and raw logits
remain private. This update publishes results only, not the local exporter
or a public self-contained reproduction package.

## All Original Measured Request Timings

Each row contains one prefill and 127 decode transitions. Warmups are omitted.
The order below groups by round and input length; round 2 actually ran input
lengths in descending order.

| Round | Input | Request | TTFT ms | ORT prefill ms | Decode wall ms | Request wall ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1 | 272.317 | 271.640 | 2116.307 | 2388.624 |
| 1 | 128 | 2 | 273.310 | 271.907 | 2118.411 | 2391.721 |
| 1 | 128 | 3 | 272.279 | 271.642 | 2119.815 | 2392.094 |
| 1 | 128 | 4 | 272.298 | 271.626 | 2118.966 | 2391.264 |
| 1 | 128 | 5 | 272.282 | 271.630 | 2116.226 | 2388.509 |
| 1 | 512 | 1 | 1059.685 | 1058.849 | 2169.877 | 3229.561 |
| 1 | 512 | 2 | 1060.144 | 1058.948 | 2174.033 | 3234.177 |
| 1 | 512 | 3 | 1059.738 | 1058.772 | 2172.248 | 3231.986 |
| 1 | 512 | 4 | 1061.456 | 1058.830 | 2171.984 | 3233.439 |
| 1 | 512 | 5 | 1059.679 | 1058.809 | 2170.950 | 3230.629 |
| 1 | 2048 | 1 | 4225.591 | 4224.602 | 2179.163 | 6404.753 |
| 1 | 2048 | 2 | 4225.208 | 4224.308 | 2178.699 | 6403.907 |
| 1 | 2048 | 3 | 4225.432 | 4224.513 | 2177.633 | 6403.065 |
| 1 | 2048 | 4 | 4225.322 | 4224.403 | 2176.183 | 6401.505 |
| 1 | 2048 | 5 | 4225.370 | 4224.533 | 2180.748 | 6406.118 |
| 1 | 4096 | 1 | 8446.789 | 8445.824 | 2203.161 | 10649.950 |
| 1 | 4096 | 2 | 8446.784 | 8445.828 | 2202.921 | 10649.705 |
| 1 | 4096 | 3 | 8447.213 | 8445.692 | 2205.154 | 10652.367 |
| 1 | 4096 | 4 | 8446.771 | 8446.047 | 2202.890 | 10649.661 |
| 1 | 4096 | 5 | 8446.208 | 8445.495 | 2202.067 | 10648.274 |
| 2 | 128 | 1 | 272.311 | 271.621 | 2100.694 | 2373.005 |
| 2 | 128 | 2 | 272.274 | 271.630 | 2100.669 | 2372.943 |
| 2 | 128 | 3 | 273.153 | 271.836 | 2101.366 | 2374.519 |
| 2 | 128 | 4 | 272.579 | 271.681 | 2100.956 | 2373.535 |
| 2 | 128 | 5 | 272.310 | 271.651 | 2104.502 | 2376.812 |
| 2 | 512 | 1 | 1059.403 | 1058.732 | 2172.049 | 3231.453 |
| 2 | 512 | 2 | 1059.988 | 1059.321 | 2177.127 | 3237.115 |
| 2 | 512 | 3 | 1059.839 | 1058.910 | 2172.711 | 3232.550 |
| 2 | 512 | 4 | 1059.429 | 1058.735 | 2171.439 | 3230.868 |
| 2 | 512 | 5 | 1059.485 | 1058.787 | 2172.146 | 3231.631 |
| 2 | 2048 | 1 | 4225.311 | 4224.027 | 2166.613 | 6391.924 |
| 2 | 2048 | 2 | 4224.538 | 4223.836 | 2163.302 | 6387.839 |
| 2 | 2048 | 3 | 4225.868 | 4224.416 | 2164.578 | 6390.446 |
| 2 | 2048 | 4 | 4225.863 | 4224.131 | 2166.082 | 6391.945 |
| 2 | 2048 | 5 | 4224.992 | 4224.071 | 2162.771 | 6387.764 |
| 2 | 4096 | 1 | 8446.663 | 8445.714 | 2201.913 | 10648.576 |
| 2 | 4096 | 2 | 8446.732 | 8445.812 | 2199.972 | 10646.704 |
| 2 | 4096 | 3 | 8446.092 | 8445.054 | 2197.405 | 10643.497 |
| 2 | 4096 | 4 | 8446.352 | 8445.373 | 2198.294 | 10644.646 |
| 2 | 4096 | 5 | 8447.148 | 8445.784 | 2198.517 | 10645.666 |

## Long-Input Extension: 32K, 64K, 128K, and 256K

Completed on 2026-10-28. Here **K = 1024**: the exact input lengths are
32768, 65536, 131072, and 262144 tokens.

### Original Single-Prefill Method: Actual Failures

Each length was actually attempted in a fresh process, with no warmup and one
requested measurement. All four failed before producing a prediction. These
are failed feasibility attempts, not completed latency/throughput measurements.
Attempt peaks cover initialization through failure, unlike the measured-request
peaks in the successful tables.

| Input | Failure stage | Failed allocation bytes | Attempt peak process GiB |
| ---: | --- | ---: | ---: |
| 32768 | Prefill, `/lm_head/MatMul` logits allocation | 26357006336 | 76.449 |
| 65536 | Prefill, first-layer `q_norm/SimplifiedLayerNormalization` | 1610612736 | 77.949 |
| 131072 | Prefill, first-layer `attn_pre_norm/SimplifiedLayerNormalization` | 1879048192 | 77.443 |
| 262144 | KV cache allocation/reset | 537131008 | 78.537 |

All errors were ORT BFCArena allocation failures. **TTFT and throughput are
N/A for this original method at every requested long length.**
The graph emits full-sequence FP32 logits of width 201088: that tensor alone
would occupy 24.546875, 49.09375, 98.1875, and 196.375 GiB respectively.
This size calculation explains a major capacity constraint; it is not a
substitute for the actual observed failure stages above.

### Separate Chunked-Prefill Method and Environment

The following successful measurements use **2048-token sequential prefill
chunks**. They are a different invocation/configuration from the original
single-prefill table and must not be interpreted as the same workload or as
an ORT implementation speedup.

- No ORT source change, rebuild, or ONNX model modification was made. The
  installed wheel and all three mapped ORT library hashes match the identities
  above; both model-file hashes also match.
- A new `jiafa-dev` container was created on an isolated Docker daemon after
  the original shared Docker storage failed. It uses
  `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel`, Python 3.12.3, the same CUDA
  13.0 runtime, and the same mounted cuDNN 9.23.2 libraries. This is an
  environment change, not a continuation in the original container.
- Physical **GPU 1**, also an NVIDIA A100-SXM4-80GB, was used instead of GPU 0.
  Its UUID was `GPU-1d6c259c-3f83-8046-c0b6-cbeb9e062c8d`. All eight formal
  workers used this UUID. Each worker checked it idle and monitored observed
  compute-PID exclusivity; clocks and power were not locked.
- CUDA provider `arena_extend_strategy=kSameAsRequested` replaces the default
  `kNextPowerOfTwo`, reducing arena growth reservation. The 64 MiB QMoE
  scratch setting and 8/1 thread counts are unchanged.
- Chunk boundaries are 2048 tokens, without overlap or truncation. All chunks
  reuse the same in-place BF16 KV buffers and cumulative position inputs.
  Local layers retain their exported 511-position sliding windows; global
  layers retain full context. Global KV capacity remains input length plus
  output predictions minus one.
- Every chunk still computes full logits for that chunk; intermediate-chunk
  logits are discarded on GPU without a CPU copy. Only the final chunk's
  last-position vector is copied, finite-checked, and argmaxed.
- TTFT now spans **all** prefill chunks and their Python/binding overhead,
  through the final CPU argmax. Prefill throughput divides total input tokens
  by the sum of synchronized ORT run times over all chunks, excluding
  inter-chunk Python/binding work. Other metric definitions remain unchanged.
- The deterministic prompt rule and forced continuation are unchanged.
  Timing has no ORT profiling, CUDA graphs, or logits-file capture.

**256K is prefill-only because of the exported model's fixed RoPE table
length, not an ONNX Runtime limitation.** The rotary tables have shape
`[262145,64]`, covering positions `0-262144`. A 256K (262144-token) prompt
occupies positions `0-262143`, and prefill produces the first prediction.
The standard benchmark then requires 127 decode transitions, consuming
positions `262144-262270`, which exceed the available table. Only one
additional decode transition fits within the existing table.

Therefore, this row makes one prediction and zero decode transitions;
decode throughput is N/A. This was a known model constraint, not an observed
decode failure. No rotary extension or model patch was used. The other three
rows make 128 predictions and 127 decode transitions.

To measure the full 127-transition decode workload without modifying the
model, a follow-up could reduce the prompt to **262018 tokens**, so the last
decode input uses position 262144. That must be labeled with its exact input
length, not as an exact 256K prompt. Alternatively, exactly 256K input tokens
plus 127 decode transitions would require a model exported with longer rotary
tables and confirmation that its positional encoding supports that range.
Neither alternative was measured in this report.

The formal protocol retains two rounds, ascending then descending lengths,
three warmups and five measured requests per process: **8 processes, 40
measured requests, 3850 predictions, and 3810 decode transitions**. The **24
warmup requests** and all feasibility/numerical probes are excluded. Each row
pools ten measured requests from two fresh processes. The matrix ran from
14:37:25 to 19:23:33 UTC on 2026-10-28.

### Pooled Chunked Results

TTFT is in **seconds**, unlike the short-input table's milliseconds.
Memory is measured-request NVML-sampled process residency, not allocation
peak or device-wide memory.

| Input tokens | Requests | TTFT p50 s | TTFT p95 s | Prefill tok/s | Decode tok/s | Peak process GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 10 | 68.207 | 68.213 | 480.522 | 52.281 | 71.928 |
| 65536 | 10 | 137.493 | 137.511 | 476.727 | 46.926 | 72.928 |
| 131072 | 10 | 279.105 | 279.206 | 469.679 | 39.709 | 74.928 |
| 262144 | 10 | 574.845 | 574.912 | 456.092 | N/A | 77.926 |

All eight formal processes and all 40 measured requests completed without OOM
or non-finite copied last-position logits. At each length, the ten requests
had one identical predicted-token stream and one identical concatenated-logits
hash. This is same-configuration repeatability, not numerical equivalence to
unchunked inference or a quality evaluation.

### Initialization and Sampling by Long-Input Process

| Round | Input | Init s | Peak process GiB | Samples | Interval p50 ms | Interval max ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32768 | 7.366 | 71.928 | 66630 | 5.286 | 7.473 |
| 2 | 32768 | 7.403 | 71.928 | 66620 | 5.288 | 9.493 |
| 1 | 65536 | 7.269 | 72.928 | 132304 | 5.284 | 8.060 |
| 2 | 65536 | 8.823 | 72.928 | 132252 | 5.288 | 6.600 |
| 1 | 131072 | 7.485 | 74.928 | 266189 | 5.288 | 10.265 |
| 2 | 131072 | 7.728 | 74.928 | 265809 | 5.296 | 10.697 |
| 1 | 262144 | 9.416 | 77.926 | 540724 | 5.288 | 55.002 |
| 2 | 262144 | 9.007 | 77.926 | 541952 | 5.289 | 13.825 |

The requested sampling interval was 5 ms, not a guaranteed maximum. In
particular, the 55.002 ms observed gap can miss brief peaks; these values
should not be treated as exact maximum allocation requirements.

### Chunking Numerical Sanity Checks and Limitations

Separate untimed probes on the replacement container compared the final
prefill logits using the same GPU, model, runtime, prompt, and
`kSameAsRequested` setting. They found differences:

| Input | Full prefill top-1 | Chunk size | Chunked top-1 | Max absolute logits difference | Logits RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 126 | 64 | 3218 | 1.455627 | 0.261406 |
| 4096 | 100 | 2048 | 100 | 1.094495 | 0.245368 |

The full-prefill top-1 margin at input 128 was 0.292189. A separate
128-prediction forced-continuation probe agreed on 127/128 predictions, with
the first prediction differing. **Chunking is therefore not claimed
numerically equivalent to unchunked prefill.** The cause of the differences
has not been established; these measurements alone cannot distinguish
floating-point/kernel-selection effects from a chunk/cache implementation
issue. Long-input full-prefill reference runs could not complete because of
the OOMs above. These tables report observed performance of this chunked
invocation, not validated numerical correctness or production readiness.

The long-input prompt hashes, runtime mappings, unrounded per-request timing
data, failure traces, and numerical probes are retained locally under
`private-model-cuda-long-results-20261028`, including `formal-summary.json` and eight
`formal-round*-input*.json` files. They are not added to this PR.

## Scope and Retained Artifacts

The raw per-process JSON includes unrounded timings, all predicted token
streams, logits hashes, prompt hashes, mapped binary hashes, GPU identity, and
sampling interval statistics. Model hashes, environment records, worker logs,
ORT profiles, and the Nsight capture/CSV are retained locally under
`private-model-cuda-results-20261027`; they are not added to this PR. The local harness
and raw captures are not a public self-contained reproduction package.

All four original short input lengths completed without OOM or non-finite last-position
logits. Two rounds and ten measured requests per length are a limited
controlled sample, not confidence intervals or production variability
coverage. No throughput ratios, memory savings against another runtime,
cross-runtime numerical differences, or task-accuracy claims are made.
