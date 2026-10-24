# Qwen3-30B-A3B INT3 Runtime Comparison

Completed on 2026-10-24 in `jiafa-dev`, using the same exported model and
matched token contexts. PR #33569 achieved **5.0290-5.1905x prefill throughput**
and **1.7509-1.9967x decode throughput** relative to PR #33555 in this experiment.
Measured NVML-sampled process peaks were **9.8418-11.0723 GiB lower**.
These are full-model request measurements, not task accuracy, numerical
equivalence, isolated-kernel speedups, or production service results.

## Runtime And Build Identity

| Runtime | Source SHA |
| --- | --- |
| PR #33555 | `d75ac16f377c0c783d289046f5cf2efe6bb5f9bc` |
| PR #33569 | `4ba6c38384cf5b12d4aa4570cd15af2461c4d515` |

Both share base `36f07f760dccb74a96d0a3c851163d6448bdfd21`.
The device was NVIDIA A100-SXM4-80GB, physical GPU 0,
UUID `GPU-739515b2-62b4-f900-86a2-50f55272a810`, driver `580.159.04`.
The GPU was checked initially idle; the sampler rejected additional or changed
observed compute PIDs. Unsampled activity cannot be excluded. Power limits and
clocks were **not locked by this experiment**.

Both source builds used Linux x86-64, Release, Ninja, CUDA 13.0, SM80,
GCC/G++ 13.3.0, NVCC V13.0.88, cuDNN 9.20.0.48, Python 3.12.3,
CMake 4.3.2, and Ninja 1.13.0.git.kitware.jobserver-pipe-1.
Compiler paths were `/usr/bin/cc`, `/usr/bin/c++`, and
`/usr/local/cuda/bin/nvcc`; C/C++/CUDA Release flags were `-O3 -DNDEBUG`.
Shared ORT libraries and Python bindings were enabled; LTO was off.

The actual caches enable `USE_FP4_QMOE`, `USE_FPA_INTB_GEMM`,
`USE_FP8_KV_CACHE`, and `USE_CUDA_NHWC_OPS`, while disabling `USE_FP8_QMOE`,
`USE_FPA_INTB_GEMM_FULL`, `USE_FLASH_ATTENTION`, and `USE_LEAN_ATTENTION`
(all names have the `onnxruntime_` prefix). NVCC and flash NVCC thread counts
were 4. Training, minimal build, and CUDA EP plugin mode were off.
Configured unit/CUDA test targets were on; skipped test execution is not proof
that those tests ran as part of these builds.

**Dependency exception:** actual ONNX source was
`ee3ccbd2b2344299d3a4506c2954a47b2181a485`, overriding the pinned submodule
`e1ac8e74b620bbdc7ba07f0440806bd93f651a28`. Nineteen other populated dependency
source directories were reused with `FETCHCONTENT_FULLY_DISCONNECTED=ON`.
Source-cache reuse is not an unrelated ORT binary/provider ABI swap. The ORT
targets were compiled/linked for the pinned sources; PR #33555's existing
clean-build directory was reconfigured in place, not deleted for this run.
Matching just the ORT SHAs without the dependency overrides is insufficient.

The measured packages were **staged Python sources plus matching build-library
symlinks, not installed wheels**. Core, pybind, shared-provider, and CUDA-provider
libraries came from each corresponding build. Package metadata collection
warnings reflect absent wheel metadata. The standalone CUDA-provider load
attempt for #33569 returned SIGSEGV; its ORT Python import and CUDA OrtValue
round-trip passed. The completed full-model runs are separate, stronger
execution evidence, not a claim that standalone loading works.

[results/build-settings.json](results/build-settings.json) records effective
cache settings, tool versions, all 20 source override paths, staged binary
hashes, and source identities. Original configure commands, caches, build logs,
validation outcomes, and staging evidence are retained under
[results/build-logs/pr33555/provenance.json](results/build-logs/pr33555/provenance.json)
and [results/build-logs/pr33569/provenance.json](results/build-logs/pr33569/provenance.json).
See [BUILD.md](BUILD.md) for reproduction instructions; a newly built wheel is
not the historically measured staged package.

## Shared Model

This reused the **October 21 export**, copied to `/tmp/onnx`; there was no fresh
full-model re-export for this runtime comparison. The source checkpoint is
`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
The recorded producer revisions are Olive
`032220e6dc7fcfc30930736784936694e7aaabe9` and Mobius
`9fb8d304cca82a245681366071c5cb8d67732021`. These are producer source identities,
not wheel identities; installed package metadata was unavailable in the retained
export environment during this archive audit.

The actual retained
[results/quantized-checkpoint-config.json](results/quantized-checkpoint-config.json)
matches symmetric RTN, group size 64, default INT4 with all expert gate/up/down
weights overridden to INT3, embedding quantization enabled, and head INT8.
The FP16 export has 48 QMoE nodes, 128 experts per layer, top-8 routing,
normalized routing weights, SwiGLU fusion/interleaving, `weights_prepacked=0`,
FP16 scales/activations/KV cache, 192 INT4 attention linears, one INT4 embedding,
and one INT8 head. This is not a uniformly three-bit model.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| Graph | 901400 | `b905062a320b9ae424bcd9cd3a0d8d54e8b626a0c9c515cf1ec6841ce9e3f045` |
| External weights | 12781146112 | `1dce48e1af84eea2a83b2d6ce138b8c9ee7ba857cf4147ab2527ad65d6038051` |

Graph plus weights total **12782047512 bytes = 12.782047512 GB =
11.904209398 GiB**, excluding tokenizer/configuration files. All packaged-file
hashes, static tensor shapes, interleave transformation chains, and external
ranges are in [results/model-audit.json](results/model-audit.json).
FC1 packed weights have shape `[128,1536,768]`, scales `[128,1536,32]`;
FC2 packed weights `[128,2048,288]`, scales `[128,2048,12]`.
No explicit bias, zero-point, or FC3 inputs were present in the audited QMoE
nodes. This does not validate unsupported explicit-zero-point or padded-tail
variants. The audit checks metadata, static transformations, and file hashes,
not packed weight contents, scale values, or isolated-layer correctness.

## Measurement Method

Batch size was 1, with input lengths 128, 512, 2048, and 4096, and 128 output
predictions per request. Each prediction stream comprises one prefill prediction
and **127 decode transitions**, not 128 decode transitions. EOS was ignored.
The pretokenized prompt is deterministic; exact IDs and hashes are in each raw
JSON. The driver checked model-file hashes, prompt hashes, forced tokens, GPU
UUID, and actually mapped runtime binary paths/hashes.

There were two alternating-order rounds (ABBA across rounds for each length),
16 sequential timing processes total, three warmups and five measured requests
per process: **80 measured requests, 10240 predictions, 10160 decode
transitions**. Each PR/length row pools ten requests from two processes.
The **48 warmup requests are extra and excluded**. Two separate numerical
reference requests and the short dispatch captures are also excluded from the
timing table. Two rounds are a small controlled comparison, not production
variability coverage or statistical confidence intervals.

PR #33569 first generated a natural-greedy 128-token reference at input length
128. Both timing runtimes then consumed that same forced token stream, also at
longer input lengths. Longer-context continuations are synthetic teacher-forced
workloads, not independent natural-greedy generation. Identical input token
contexts do **not** guarantee identical expert routing: floating-point
differences can change router decisions.

CUDA graphs, ORT profiling, and debug switches were disabled for timing. CUDA
was primary and required to load; CPU shape operations remained allowed.
Intra/inter-op thread counts were 8/1. The QMoE scratch setting was
`ep.cuda.qmoe_int_dequant_max_scratch_bytes=67108864` (64 MiB per operation),
not a process-memory cap. Fresh empty GPU KV caches were used per request.

- TTFT is pretokenized prompt to CPU argmax, excluding session initialization
  and KV reset; it includes last-position logits transfer/finite check/argmax.
- Prefill throughput is total input tokens divided by total synchronized ORT
  prefill-run time, not TTFT-based throughput. Full-sequence logits stay on GPU;
  the final-position vector is copied to CPU.
- Decode throughput is total 127-step transitions divided by total decode wall
  time, including Python, binding, masks, GPU execution, last-position D2H,
  finite checks, argmax, and logits hashing.
- Memory is NVML-sampled process residency during measured requests, requested
  sampling interval 5 ms. The table takes the larger process peak across the
  two processes per row. It is neither an exact allocation peak nor device-wide
  usage; raw vectors, device/lifetime peaks, and actual sampling gaps are retained.

## Measured Results

TTFT values are pooled p50/p95 in milliseconds; throughputs are tokens/second;
peak is measured sampled process GiB. Values are rounded only for display.

| Input | PR | TTFT p50 ms | TTFT p95 ms | Prefill tok/s | Decode tok/s | Peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 33555 | 1080.192 | 1080.545 | 118.677 | 50.179 | 24.412 |
| 128 | 33569 | 216.055 | 216.111 | 596.825 | 100.194 | 14.570 |
| 512 | 33555 | 4272.235 | 4272.591 | 119.897 | 49.325 | 24.436 |
| 512 | 33569 | 826.965 | 827.786 | 620.321 | 97.486 | 14.436 |
| 2048 | 33555 | 17044.647 | 17045.677 | 120.170 | 46.744 | 26.539 |
| 2048 | 33569 | 3285.147 | 3285.503 | 623.745 | 87.082 | 15.467 |
| 4096 | 33555 | 34151.688 | 34154.682 | 119.944 | 43.430 | 32.469 |
| 4096 | 33569 | 6636.490 | 6636.725 | 617.369 | 76.040 | 21.461 |

| Input | Prefill ratio #33569/#33555 | Decode ratio #33569/#33555 | Process peak saving GiB |
| ---: | ---: | ---: | ---: |
| 128 | 5.0290 | 1.9967 | 9.8418 |
| 512 | 5.1738 | 1.9764 | 10.0000 |
| 2048 | 5.1905 | 1.8630 | 11.0723 |
| 4096 | 5.1472 | 1.7509 | 11.0078 |

Session initialization is reported separately below, as the minimum/maximum
across two processes in seconds. This includes complete ORT session construction
and associated graph work, loading, allocation, and prepacking; **it is not a
prepack-only timer** and is not included in TTFT or steady-state throughput.

| Input | PR | Init min s | Init max s |
| ---: | ---: | ---: | ---: |
| 128 | 33555 | 37.549 | 37.692 |
| 128 | 33569 | 36.299 | 36.725 |
| 512 | 33555 | 37.442 | 37.548 |
| 512 | 33569 | 36.577 | 36.693 |
| 2048 | 33555 | 37.645 | 37.650 |
| 2048 | 33569 | 36.623 | 36.660 |
| 4096 | 33555 | 37.619 | 37.744 |
| 4096 | 33569 | 36.595 | 36.692 |

Unrounded aggregate and per-round values are in
[results/summary.json](results/summary.json); the completed process commands,
environments, identities, and original output hashes are in
[results/manifest.json](results/manifest.json).

## Numerical Check

Only input length 128 has full last-position logits captured for all 128
predictions, each vector of width 151936 and dtype FP16. The PR #33569 natural
reference and PR #33555 on the same forced continuation are finite and have
**128/128 top-1 agreement**, but **allclose is false** with `atol=rtol=0.01`:
maximum absolute difference **0.5625**, mean absolute difference
**0.01425221532938465**, RMSE **0.02311116403843693**.

The paired timing trials additionally agree on 1280/1280 predicted top-1 tokens
per input length (5120/5120 total paired predictions), derived from the raw
trial token lists. That is a token-level observation, not an all-length
full-logits comparison. Neither check establishes numerical equivalence,
quantization quality, downstream task accuracy, or an individual-kernel bug.
See [results/logits-comparison.json](results/logits-comparison.json),
[results/reference-tokens.json](results/reference-tokens.json),
[results/reference33555.json](results/reference33555.json), and
[results/reference33569.json](results/reference33569.json).

## Dispatch Evidence

The existing [results/dispatch/summary.json](results/dispatch/summary.json) and
its two kernel-summary CSVs are preserved. Separate short captures cover the
full process: initialization, 128-token prefill, and one decode step, with no
warmup. PR #33555 shows 96 `FusedFc1`, 96 `Fc2AndFinalize`, and 96
`PackBitplanes` launches. PR #33569 shows 96 each of `FusedFc1FastKernel`,
`Fc2RouteFastKernel`, and `FinalizeRoutesFastKernel`, and no `PackBitplanes`.
The compute-family counts correspond to 48 layers times two inference passes.
The 96 packing launches are a separate aggregate observation, not evidence of
packing on every inference pass.

These are **not phase-exclusive counts**; packing presence is consistent with
prepacking, but aggregate statistics do not locate its phase. Profiler overhead,
the short capture, and unseparated initialization/prefill/decode prohibit
attributing the full-model speedup to an isolated kernel or interpreting trace
timings as the benchmark results.

## Implementation Method Comparison

This comparison describes the **INT3 paths at the pinned revisions**, not every
feature in either PR. Both consume the same canonical, row-local packed INT3
model with FP16 scales/activations and perform floating-point dot products after
unpacking/dequantization. Neither observed path is an INT3 Tensor Core GEMM.

Source references:

- #33555: [INT3 compute and bit-plane kernels](https://github.com/microsoft/onnxruntime/blob/d75ac16f377c0c783d289046f5cf2efe6bb5f9bc/onnxruntime/contrib_ops/cuda/moe/qmoe_int3.cu)
  and [host prepack/compute](https://github.com/microsoft/onnxruntime/blob/d75ac16f377c0c783d289046f5cf2efe6bb5f9bc/onnxruntime/contrib_ops/cuda/moe/qmoe_int3.cc).
- #33569: [scalar/fast INT3 kernels and dispatch](https://github.com/microsoft/onnxruntime/blob/4ba6c38384cf5b12d4aa4570cd15af2461c4d515/onnxruntime/contrib_ops/cuda/moe/qmoe_kernels.cu)
  and [host validation/allocation](https://github.com/microsoft/onnxruntime/blob/4ba6c38384cf5b12d4aa4570cd15af2461c4d515/onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc).

| Aspect | #33555 observed INT3 path | #33569 observed INT3 fast path |
| --- | --- | --- |
| Weight representation | Session-time `PrePackInt3` converts constant weights to a private K32 bit-plane layout; model bytes stay unchanged. | Reads canonical packed bytes directly; no INT3 bit-plane prepack. Its separate offline SM80 prepack support for INT2/INT4 is **not** the INT3 method measured here. |
| Weight decode | `Weight<T, Bits>` reconstructs one code/weight at a time inside the feature loop. | `Unpack8CenteredCodes` / `Unpack8CenteredCodes128` extract eight centered codes at a time; INT3 uses word loads and funnel shifts. |
| FC1 gate/up + activation | `FusedFc1`: one warp per route/intermediate-column item, four items per 128-thread block. | `FusedFc1FastKernel`: 256-thread CTA per route/output tile, two adjacent intermediate columns per warp (16 per CTA). |
| Input reuse | Each dot-product loop reads its input elements from global memory; hardware caches may serve those reads. No explicit shared-input staging in these kernels. | Each CTA stages its route input in shared FP32 memory, then reuses it across columns and gate/up dot products with `float4` shared-memory reads. |
| FC2 and routing | `Fc2AndFinalize`: one warp per token/output column; loops over the top-k routes and performs each route dot product before accumulating its weighted result. | `Fc2RouteFastKernel`: routes are separate grid entries; each half-warp computes two output columns (32 per CTA), using shared activations. `FinalizeRoutesFastKernel` subsequently sums weighted route results. |
| Intermediate storage | Fused FC2/finalization avoids a separate route-output tensor. | Adds `fc2_partial`, stored in activation dtype, with `rows * top_k * hidden` elements, plus an extra finalization launch. |
| Specialization | Uses the scalar dot-product organization for this workload. FC1 routes are already parallel; this is not an entirely serial implementation. | Fast path requires aligned dimensions/pointers, absent explicit zero points, bounded grid/shared-memory sizes, and route scratch; otherwise uses scalar fallback. |

For this model, hidden size 2048 is divisible by 256, intermediate size 768 is
divisible by 128, and block size 64 is divisible by 8. There are no explicit zero
points. FC1/FC2 shared allocations are respectively 8 KiB and 3 KiB per CTA.
Prefill has `rows * top_k = input_length * 8` routes, at most 32768 for input
4096; decode has eight routes. These fit the fast-path grid bounds. The short
trace confirms fast dispatch at input128 and one decode step; the dimensions
also satisfy the source predicates at the longer measured lengths, without
claiming separate traces were captured for every length.

## Why #33569 Is Faster Here

The source provides a plausible architectural explanation for the measured
advantage, and the trace confirms the relevant kernel families were executed.
It does **not** quantify the contribution of each optimization:

1. **Amortized unpacking and better input reuse.** Eight-code unpacking and
  adjacent-column computation let #33569 reuse input values and do less
  repeated per-weight address/decode work. Cooperative shared-memory staging
  makes input reuse explicit within each output tile. #33555 instead repeats
  scalar weight reconstruction and input reads across dot products. This is
  an instruction/reuse advantage, not proof of eightfold fewer instructions
  or a measured reduction in DRAM traffic: cache hits and compiler behavior
  matter, and no instruction/bandwidth counters were collected.
2. **More parallel FC2 work, especially for decode.** In #33555, the expensive
  dot products for eight selected experts sit inside each token/column warp's
  route loop. #33569 exposes those route dot products as independent CTAs and
  leaves only the inexpensive weighted sum to finalization. At one decode
  token, the source launches 512 FC2 route CTAs (`2048 / 32 * 8`) instead of
  512 scalar FC2 CTAs (`2048 / 4`), but they have different work and thread
  organization: the fast CTAs each handle one route and 32 columns, not all
  eight routes and four columns. The benefit is the work decomposition and
  reuse, **not simply more CTAs**. Additional scratch traffic and a third
  kernel launch are real costs that the measured implementation nevertheless
  overcomes on this workload.
3. **A favorable aligned model shape.** This model meets the fast-path
  constraints throughout the tested lengths. The result should not be
  extrapolated to shapes or explicit-zero-point configurations that use the
  fallback, or to other GPUs without measurement. Both paths parallelize
  FC1 routes; route parallelism is not unique to #33569 in FC1.
4. **Full-model costs limit the observed ratio.** QMoE improvements are diluted
  by attention, KV-cache work, other operators, and request overhead. The
  decode advantage falls from about 2.00x to 1.75x as context grows, consistent
  with a larger share of context-dependent non-QMoE work. This is a plausible
  explanation, not a measured per-operator attribution. The approximately
  5.03-5.19x prefill ratio likewise remains a full-model result, not a claimed
  speedup for any individual kernel.

**Prepacking and memory need separate interpretation.** #33555's prepacking
happens during session setup, outside the steady-state timing. Merely avoiding
that setup work cannot explain #33569's prefill/decode speedup; the representation
and compute organization used afterward can matter. Its bit-plane allocation
has the same packed byte size as the input weights, not an FP16-expanded copy.
#33569 avoids that private conversion but adds route scratch: for FP16 at
input4096, `4096 * 8 * 2048 * 2` bytes = 128 MiB per FC2 partial tensor, not a
persistent allocation for every layer. The scratch limit documented above is
not a cap on all such allocations. Neither observation alone explains the
9.84-11.07 GiB sampled residency gap. Persistent initializer ownership, graph
transformations, allocator retention, and workspaces would need an allocation
breakdown before assigning that gap to one mechanism.

Thus **#33569 is the better-performing implementation for this measured
workload**, with source-supported reasons to prefer its INT3 compute
organization. This does not establish numerical superiority: in particular,
its FP16 route-partial materialization and different reduction order can change
rounding. They are possible contributors to the logits differences, not a
diagnosis of their source. Phase-separated profiling, hardware counters,
allocation tracing, and controlled optimization ablations would be needed for
causal attribution; quality evaluation remains separate.

## Archived Evidence

All 16 raw timing JSONs, including sampled-memory vectors, are retained losslessly
in [results/raw-timings.tar.gz](results/raw-timings.tar.gz): **28462988 bytes
uncompressed**, **2637497 bytes compressed**. Member sizes/hashes are in
[results/raw-timings-index.json](results/raw-timings-index.json). Archive SHA256:
`e62881a2b03543882a9a667f2e0f66c9b59319f0bedaace66b99a143a39abc88`.
Small measurement JSONs and all measurement/build logs are directly readable
under `results/`; no individual new recipe artifact exceeds 10 MB.

The two raw NPZs total **77794340 bytes** and binary Nsight reports/SQLite
are intentionally not included in recipe results. A persistent external archive
at `/datadisks/disk1/jiafa/accuracy/int3_qmoe/results/pr33555-pr33569-qwen-2026-10-24.tar.gz`
contains all original measurement outputs and complete dispatch artifacts:
**56 files, 111208103 uncompressed file bytes, 69606336 archive bytes**.
Its SHA256 is
`4f877f0cbe20acbfc2078ee5f8a314bd0b63611bc47e33069aafbf168eb66a7a`.
[results/external-archive.json](results/external-archive.json) records its complete
member inventory and hashes. Absolute container paths in original JSONs are
historical provenance, not portable artifact locations.

Published CSV/cache text copies normalize CRLF and trailing empty lines for
repository whitespace checks. Historical build-helper scripts are stored as
`.py.txt`, not executable recipe sources. Original capture hashes refer to
the unnormalized originals; the external archive retains original captures.

[results/experiment-source-hashes.json](results/experiment-source-hashes.json)
records `benchmark.py`, `run_comparison.py`, and `compare_logits.py` at their
executed paths, captured after completion **before formatting**. No experiment
script was changed by this archival task. Later formatting must occur only
after this capture and must not replace these recorded experiment hashes with
post-format hashes or imply the measurements were rerun.

Archive integrity and report values were checked against the original JSONs;
[results/report-validation.json](results/report-validation.json) records the
executable validation outcome, counts, and source/report hashes. No new model
export, performance measurement, or accuracy evaluation was run for this report.