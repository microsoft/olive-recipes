# Qwen3-30B-A3B Block64 Performance Analysis

Measured October 4, 2026, following the direct-ORT timing methodology in
[ONNX Runtime PR #33005](https://github.com/microsoft/onnxruntime/pull/33005).
This is a synthetic batch-one full-model comparison, not a quality benchmark
or an isolated INT2-versus-INT4 kernel experiment.

A Chinese discussion of llama.cpp implementation differences and the follow-up
experiments is available in [discussion.md](discussion.md).

## Environment and Models

- NVIDIA A100-SXM4-80GB, physical GPU 1, SM80; Docker `jiafa-dev`.
- GPU UUID: `GPU-1d6c259c-3f83-8046-c0b6-cbeb9e062c8d`.
- Clean CUDA ORT wheel, Release `main@62ac19abcf`, containing PR #33005.
- CUDA runtime loaded from CUDA 12.8; cuDNN libraries from the 9.7 directory.
- Same pinned Qwen checkpoint, byte-identical tokenizer, FP16 graph/activations,
  symmetric block64 quantization, INT4 attention/embeddings and INT8 head.
- INT4: all expert projections INT4. Mixed: gate/up INT2 in the 24 layers
  listed in the recipe README, with all remaining expert projections INT4.
- Qwen has 48 QMoE layers, top-k 8, 4 KV heads with head size 128,
  full-history KV, and vocabulary size 151936.

Local model directories under `/sunghcho_data/jiafa/qwen3-30b-a3b/`:

```text
rtn_manual_fp16_int4_lmhead_int8_block64_perf_20261004/
rtn_manual_fp16_complement_int2_int4_block64_perf_20261004/
```

## Dispatch Validation

Separate debug runs used default dispatch switches unset and
`ep.cuda.qmoe_int_dequant_max_scratch_bytes=1`. This guard blocks full dense
expert-weight materialization, not all temporary allocations.

At 512 input tokens and four output tokens, mixed execution recorded:

- 24 `packed_int_prefill` calls at 512 rows and 24 `grouped_moe` calls for
  the remaining INT4 layers.
- 72 `packed_int_gemv` calls at one row: 24 mixed layers times three decode
  steps. Remaining INT4 layers recorded 72 `grouped_moe` calls.
- INT4 baseline: 48 `grouped_moe` calls at 512 rows and 144 at one row.

A separate 4096-token mixed run recorded 24 `packed_int_prefill` calls,
24 remaining-layer `grouped_moe` calls and 24 single-token packed GEMV calls.
Both checks completed with finite last-position logits and the one-byte guard.
These are full-model path checks, not independent numerical-reference tests.

## Measurement Method

Batch size 1; input lengths 128/512/2048/4096; exactly 128 output tokens;
three warmups and ten measured requests per model/length. Each of the eight
configurations ran in a fresh process on the same GPU. Model order was
INT4/mixed, mixed/INT4, INT4/mixed, mixed/INT4 across increasing input lengths.
This changes configuration order, but is not repeated interleaved A/B trials.
Each model completed 40 measured requests and 5120 generated tokens.
All ten generated token sequences matched within each configuration.

The prompt repeats `Paris is the capital of France. Its museums and public
parks attract visitors.`, followed by `Summarize the context in detail.
/no_think`, using Qwen chat delimiters and an empty completed thinking prefix.
Repeated context tokens are truncated to obtain exact input lengths.
Greedy generation ignores EOS to enforce a fixed output length.

- TTFT: pretokenized input and empty GPU KV ready until the first greedy token
  is available on CPU. Includes binding, synchronized prefill, last-position
  logits D2H, finite check and CPU argmax. Excludes loading, tokenization,
  initial KV reset, queuing and networking; not service-level TTFT.
- Prefill TPS: total input tokens divided by summed synchronized ORT prefill
  times. Excludes binding setup and token selection. Full-sequence logits
  remain computed by the original graph.
- Decode TPS: 1270 tokens divided by summed decode times for ten requests,
  excluding the first token produced by prefill. Includes Python orchestration,
  binding, GPU execution, last-position logits D2H, finite check and argmax.
- Memory: NVML process-resident bytes including weights, KV, workspace and
  retained arena allocations. Requested sampling interval 5 ms; observed maximum
  gaps ranged from 12.35 to 15.85 ms. Peaks may miss brief allocations and are
  not exact allocator high-water marks.
- Profiling and route logging disabled during timing. KV stays on GPU;
  only last-position logits are transferred. No CUDA graph capture is used.
- INT2 prefill and GEMV switches left at defaults. The one-byte dense scratch
  guard is retained for both models. It is a validation control, not required
  for normal execution.
- A process guard required the benchmark GPU to be free before loading and
  rejected another compute process appearing during sampling. GPU 0 was
  occupied; other GPUs on this shared host were not controlled.

## Results

| Input Tokens | Model | TTFT P50 (ms) | TTFT P95 (ms) | Prefill TPS | Decode TPS | Process Peak (GiB) |
|---:|---|---:|---:|---:|---:|---:|
| 128 | INT4 | 31.01 | 31.05 | 4350.57 | 110.28 | 31.80 |
| 128 | Mixed | 29.26 | 29.30 | 4618.67 | 110.64 | 26.53 |
| 512 | INT4 | 61.39 | 61.46 | 8560.79 | 108.69 | 31.78 |
| 512 | Mixed | 55.41 | 55.58 | 9535.26 | 108.08 | 26.69 |
| 2048 | INT4 | 172.92 | 173.02 | 11958.53 | 100.48 | 31.91 |
| 2048 | Mixed | 156.42 | 157.14 | 13239.81 | 98.97 | 27.69 |
| 4096 | INT4 | 334.90 | 335.26 | 12294.33 | 89.12 | 33.82 |
| 4096 | Mixed | 309.55 | 312.22 | 13325.99 | 87.49 | 29.59 |

Changes below were calculated from unrounded timings and throughput totals.

| Input Tokens | TTFT P50 Reduction | Prefill TPS Gain | Decode TPS Change | Process Peak Saving (GiB) |
|---:|---:|---:|---:|---:|
| 128 | 5.65% | 6.16% | +0.33% | 5.27 |
| 512 | 9.74% | 11.38% | -0.57% | 5.09 |
| 2048 | 9.54% | 10.71% | -1.49% | 4.23 |
| 4096 | 7.57% | 8.39% | -1.84% | 4.22 |

## Findings and Limits

Mixed prefill was faster for all four lengths, with lower TTFT P50. Decode
was essentially unchanged at 128 and slightly slower at longer lengths;
controlled repeated runs are required before claiming a stable regression.
Mixed sampled process peak residency was lower in this run; this is not
an exact allocation attribution or a guaranteed memory reduction.

This comparison holds quantization group size constant, unlike the historical
GPT-OSS comparison in PR #33005. However, different greedy continuations can
change MoE routing, and mixed and INT4 use different dispatch paths. Results
do not establish isolated kernel speedups, accuracy preservation, task success,
real-service throughput, BF16 coverage, concurrency or CUDA capture safety.
P95 is based on only ten requests. These performance runs did not evaluate
accuracy; separate full-test ARC-Easy and preliminary quality checks are documented in
[ACCURACY.md](ACCURACY.md).

## INT8-Activation INT2 GEMV Experiment

Measured October 7, 2026, on A100-SXM4-80GB GPU 1 in Docker `jiafa-dev`.
This is a standalone synthetic kernel experiment, separate from the full-model
INT4/mixed comparison above. Here **baseline means INT2 weights with FP16
activations**, not the all-INT4 model. The experimental ORT branch
`qmoe-int2-int8-activation-decode` is based on `main@98468bff47`; its uncommitted
prototype was compiled with CUDA 12.8 for SM80, reusing an existing CUDA weight
preprocessing object. This is not a new ORT wheel or full provider build.

The default-off `ORT_QMOE_INT2_INT8_ACTIVATIONS=1` prototype quantizes each
thread's 16-element activation tile dynamically to symmetric INT8, packs four
values per DP4A operand, accumulates integer dot products in INT32, and restores
activation and blockwise weight scales into FP32 accumulation. This is local
tile quantization, not per-row INT8 quantization or INT8 Tensor Core GEMM.
Weights remain packed INT2 in device memory; no full INT8 weight buffer is
materialized. Default dispatch is unchanged, and INT4 projections and grouped
GEMM/prefill are not modified by this experiment.

Two experimental implementations were compared:

- **Float-unpack prototype:** the existing converter decodes INT2 to FP16,
  then the prototype converts those values to integers for DP4A.
- **Direct-integer prototype:** integer shifts/masks decode the existing
  pair-interleaved INT2 codes as `code - 2`, preserving the layout mapper,
  without an intermediate floating-point weight conversion.

The test calls the ordinary symmetric, non-fused GEMV with eight expert rows,
one row per expert, deterministic synthetic weights, blockwise scales and
sinusoidal FP16 activations. It does not invoke fused SwiGLU, FC2 finalize or
split-K, so disabling experimental fused split-K does not confound this test.
Each path uses 20 warmups and 200 timed launches, measured by CUDA events on
one stream. Timing includes activation quantization inside the kernel but
excludes allocation, H2D/D2H transfers, weight preprocessing and CPU reference
checks. Baseline runs before the experimental mode; the old and new binaries
run sequentially, not randomized alternating trials. Repeated data may remain
cached. These are kernel latencies, **not end-to-end decode TPS**.

| N | K | Block | INT2/FP16 Baseline (us) | Float-Unpack INT8 (us) | Direct-Integer INT8 (us) |
|---:|---:|---:|---:|---:|---:|
| 1536 | 2048 | 64 | 15.2474 | 29.4349 | 23.5264 |
| 1536 | 2048 | 128 | 15.1245 | 29.2198 | 23.4138 |
| 2048 | 768 | 64 | 9.84576 | 19.5277 | 14.2490 |
| 512 | 512 | 64, zero input | 6.89664 | 10.7622 | 9.73824 |

The baseline column is from the direct-integer binary's run; the preceding
float-unpack binary measured baseline latencies of 15.2986, 14.9811, 9.67168
and 6.75328 us, respectively. An earlier direct-integer repeat measured
23.5213, 23.3370 and 14.0902 us for the three nonzero cases, consistent with
the direction of the result but not a statistical confidence interval.

Direct integer unpacking reduces the nonzero-input prototype latency by about
20%-27% relative to float unpacking. It still takes about **1.45-1.55 times
the baseline latency**. Thus the current prototype does not demonstrate an
INT8-activation speedup on A100. The experiment identifies avoidable weight
conversion overhead, but does not establish which remaining costs dominate;
activation quantization, operand packing and register pressure need profiling.
It does not rule out a different INT8 implementation or different hardware.

| N x K / Block | Baseline vs CPU Max Absolute Error | INT8 vs Baseline Relative L2 | INT8 vs Baseline Max Absolute Difference |
|---|---:|---:|---:|
| 1536 x 2048 / 64 | 0.000235558 | 2.38305% | 0.0285034 |
| 1536 x 2048 / 128 | 0.000372648 | 2.53564% | 0.0276985 |
| 2048 x 768 / 64 | 0.000236630 | 2.20280% | 0.0239258 |

All outputs were finite; zero input produced exact zero in both modes. Full
output fingerprints match between float-unpack and direct-integer prototypes
for all four cases, supporting equivalence on these inputs. Fingerprints and
error statistics do not prove correctness for arbitrary inputs. The CPU
reference checks the original unquantized-activation dot product; an independent
reference for the prototype's tilewise activation quantization was not included
in this initial run; the follow-up below adds that check.
The 2.20%-2.54% relative L2 difference is an output error, **not a task-accuracy
drop**. No model quality evaluation was run for this path.

Remaining qualification includes BF16 execution, fused SwiGLU/FC2 paths,
broader input tests, full ORT integration,
profiling, end-to-end model decode and target-hardware runs. No H200 or Spark
results, model decode-TPS gains, memory savings or production readiness are
claimed. The prototype is local and has not been submitted as an ORT PR.

### Follow-Up: Prequantized Input Reuse and Direct DP4A Packing

Also measured October 7, 2026, in `jiafa-dev` on A100 GPU 1 using CUDA 12.8
and SM80. A separate experimental entry point quantizes each source input row
once into INT8 values and FP32 scales, retaining the same 16-element groups
and rounding rule as the online prototype. GEMV then reuses these values across
output columns and mapped expert rows. This is not llama.cpp's block32 Q8_1
format. Tests cover eight distinct source rows with a reversed row mapping,
and one shared source row consumed by eight experts.

The subsequent weight-side optimization replaces intermediate INT16 decoding,
layout reordering and repacking with `__byte_perm`, shifts/masks and `__vsub4`
to construct signed four-byte DP4A operands directly. Packed device weights,
activation quantization, FP32 accumulation and thread configuration remain
unchanged. An initial scalar direct-packing variant was slower; the following
results are for the byte-parallel implementation, not that scalar variant.
Default ORT dispatch is unchanged.

The extended harness measures four modes: INT2/FP16 baseline, online INT8,
prequantization plus GEMV, and GEMV with already-quantized inputs. Each mode
uses 20 warmups and 200 timed invocations; mode order rotates over three rounds
and the reported value is the median. CUDA Graph timing captures 200 invocations
and divides elapsed device-event time by 200. The total mode includes a
quantization kernel for every GEMV invocation, including temporary-buffer writes
and reads; GEMV-only excludes that quantization. Allocation, weight preprocessing,
transfers and references are excluded. Before/after binaries were run sequentially
in two alternating pairs, not randomized trials or a statistical confidence study.

The table uses the second pair's CUDA Graph results for **one shared input row,
eight expert rows**. Baseline values are from the after binary. These are
standalone, non-fused synthetic GEMV latencies, not complete QMoE or decode TPS.

| N | K | Block | INT2/FP16 Baseline (us) | Online INT8 Before / After (us) | Prequantized GEMV Before / After (us) | Prequantization + GEMV Before / After (us) |
|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 2048 | 64 | 10.8851 | 17.6794 / 12.0269 | 16.0358 / 10.3782 | 18.9235 / 13.1942 |
| 1536 | 2048 | 128 | 10.7571 | 17.5872 / 12.0115 | 16.0102 / 10.3475 | 18.9235 / 13.2762 |
| 2048 | 768 | 64 | 6.55872 | 10.2451 / 7.49568 | 9.40032 / 6.70208 | 12.0166 / 9.31328 |

Byte-parallel packing reduces online INT8 latency by approximately 27%-32%
and prequantized GEMV-only latency by approximately 29%-35% on these cases.
GEMV-only is slightly faster than the baseline on the two K=2048 cases, but
**including quantization remains slower than INT2/FP16 on all three cases**.
The result does not demonstrate end-to-end model speedup. Timing from this
Graph experiment should not be directly compared with the initial non-Graph
table above, which also used a different source-row arrangement.

Compiler resource reports for the ordinary FP16, no-bias experimental kernels
show registers decreasing from 96 to 79 with online quantization, and from 96
to 56 with prequantized inputs, for block64 and block128. Stack and local
storage were zero before and after: this is not evidence of eliminating spills.
Nsight Compute hardware-counter collection failed with `ERR_NVGPUCTRPERM`;
no bandwidth, occupancy or stall-counter attribution is claimed.

Validation passed for all eight shape/input-mapping cases, including zero
inputs. An independent CPU reference checks activation scales, round-to-nearest
INT8 values and the quantized dot product. Maximum absolute quantized-reference
error was below 0.000396; zero inputs remained exact zero. Before/after online
output fingerprints matched for all eight cases, prequantized and online outputs
were identical on these cases, and CUDA Graph outputs matched ordinary launches.
The packed mapping additionally passed 1,000 random 16-byte tile checks covering
all 64 decoded elements. Compute Sanitizer memcheck reported zero errors.
These checks do not establish arbitrary-input correctness, model quality, BF16
runtime behavior, fused SwiGLU/FC2 correctness or full ORT integration. Activation
32-bit loading, thread/ reduction changes and upstream quantization fusion were
not part of this weight-packing change.

### Follow-Up: Fused FC1 Integration and Full-Model A/B

Measured October 7, 2026, on A100-SXM4-80GB physical GPU 6 in Docker
`jiafa-dev`. Unlike the standalone experiments above, this follow-up builds
the complete CUDA provider, matching Python binding and shared-provider library
from the local experimental branch based on `main@98468bff47`. The build contains
uncommitted experimental changes; the commit ID alone does not reproduce them.
The previously validated `main@62ac19abcf` wheel was not overwritten or combined
with the new provider. This remains a local prototype, not a submitted ORT PR
or functionality delivered by this recipe.

#### Implementation

The default-off `ORT_QMOE_INT2_PREQUANTIZED_FC1=1` path is integrated at the
packed INT2 QMoE FC1 dispatch in `moe_quantization.cc`. It currently applies to
FP16 activations, no FC1 bias, and symmetric block64/block128 weights. Other
cases retain the existing launch path. In the eligible branch:

1. Quantize each original source row once into symmetric INT8, with one FP32
  scale per 16 activation values: scale is amax/127, values use round-to-nearest
  and clamp to [-127, 127]. Stream-aware ORT scratch buffers hold values/scales.
2. Reuse that row across selected experts and output columns in the fused
  gate/up GEMV and SwiGLU epilogue, rather than quantizing it in each dot tile.
3. Decode packed uniform INT2 weights directly into four signed bytes using
  byte permutation, masks and byte subtraction, then accumulate with DP4A
  into INT32. Prequantized activation operands use aligned 32-bit loads.
4. Restore activation scales into FP32 accumulation and factor out the shared
  weight scale within each tile. This changes floating-point association;
  bitwise equivalence to the preceding implementation is not guaranteed.

Weights remain INT2 in device memory; there is no full INT8 weight expansion.
This is not an INT2/INT8 Tensor Core implementation. The existing source-row
mapping, activation epilogue, FC2 and finalization remain in use. Quantization
is still a separate kernel, not fused into the upstream producer. The earlier
online switch `ORT_QMOE_INT2_INT8_ACTIVATIONS` is disabled in both A/B arms.

For comparison, the inspected llama.cpp revision
`c479922ac520a08969b4c1dc154d7bbb3c386d85` also uses a separate activation
quantizer and pooled temporary storage for CUDA MMVQ, reusing Q8_1 activations
across columns and applicable shared experts; eligible indexed MMVQ paths can
fuse up/gate GLU. Its Q8_1 groups contain 32 values, while this prototype uses
16-value groups. Its IQ2 formats use codebooks/sign decoding, unlike ORT's
uniform symmetric INT2 values [-2, -1, 0, 1]. These are implementation parallels,
not format equivalence or a llama.cpp-versus-ORT performance comparison. No
same-model cross-runtime benchmark or upstream-producer quantization fusion
has been established here.

#### Model Measurement and Dispatch

Both arms use the same mixed model directory listed above, the same newly
built runtime, GPU 6, batch size 1, 128 input tokens and exactly 16 output tokens.
Each process has two warmups and five measured requests. The first pair runs
default then prequantized; the second pair reverses that order. Each arm thus
has ten measured requests and 150 timed decode tokens, excluding prefill's
first token. Timing retains the binding, synchronized ORT execution, last-row
D2H, finite check and CPU argmax methodology above. Profiling and route logging
are disabled during timing; the one-byte dense-dequantization scratch guard
is retained. Other GPUs on this shared host are not controlled.

Here **baseline means the same mixed INT2 model with the new switch disabled**,
not the all-INT4 model or the earlier wheel. Absolute TPS should not be treated
as a controlled comparison with the October 4 results above.

| Order | Default Mixed Decode TPS | Prequantized FC1 Decode TPS | Default ORT Run P50 (ms) | Prequantized ORT Run P50 (ms) | Default Process Peak (GiB) | Prequantized Process Peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Default, then prequantized | 111.6133 | 112.7932 | 7.3397 | 7.1691 | 26.5977 | 26.5352 |
| Prequantized, then default | 111.1562 | 112.9433 | 7.2548 | 7.2609 | 26.5391 | 26.5312 |
| Combined TPS; maximum observed peak across both orders | 111.3843 | 112.8682 | N/A | N/A | 26.5977 | 26.5352 |

Process Peak is the existing raw-result field `measurement_peak_process_gib`:
NVML-sampled GPU memory resident for the benchmark process during the measurement
window, after warmups, in GiB (bytes / 2^30). Sampling requests a 5 ms interval;
it is not an exact ORT allocator peak, kernel workspace size, or whole-device
memory usage. Per-order rows show that process's sampled peak; the combined row
takes the maximum of the two order peaks for each arm, not their mean or median.
The small differences do not establish a memory reduction from the INT8 path;
allocator residency and sampling variability can obscure its added scratch.

Combined decode TPS improves **1.33%**. Both pairs have the same TPS direction,
but the second pair's ORT-run median is not improved. This is a small observed
gain, without a statistical confidence interval, not evidence of a robust
general speedup. All 20 measured requests produce the same 16-token sequence
across both arms, and last-position logits are finite. Token agreement on this
single synthetic prompt does not establish accuracy preservation.

A separate Nsight Systems capture covers 15 measured decode steps and confirms
360 activation-quantization launches and 360 prequantized INT2 fused FC1
launches: 24 eligible layers per step. The profiled mean durations are about
3.52 us for quantization and 10.89 us for fused INT2 FC1. These establish actual
dispatch and expose the extra launch cost; profiled times are not used for TPS
and are not directly comparable with earlier captures or standalone timings.

The fused standalone harness additionally checked the independent quantized
CPU reference, shared and two-source-row mappings, default/custom activation
parameters and Compute Sanitizer memcheck. These checks do not substitute for
full QMoE numerical qualification. Remaining work includes reducing the
separate quantization/launch overhead, broader sequence and routing shapes,
BF16/bias/fallback tests, CUDA capture/allocation lifetime validation and full
task-accuracy evaluation with the experimental switch enabled. Existing recipe
accuracy results do not qualify this newly quantized-activation path. Default
dispatch remains unchanged.

## FP16-Activation INT2 Fused FC1 CTA Experiment

Measured October 8, 2026, in Docker `jiafa-dev` on A100-SXM4-80GB GPU 7,
using the local experimental ORT branch based on `main@98468bff47` and a
rebuilt CUDA provider with matching Python binding/shared-provider library.
Both arms retain FP16 activations, packed INT2 weights and FP32 accumulation.
This is a separate experiment from INT8 activation quantization above.

The default INT2 specialization already uses CTA4 (`kCtaN / 2`, where
`kCtaN = 8`). The default-off `ORT_QMOE_INT2_FP16_FC1_CTA8=1` prototype changes
the fused FC1 tile to CTA8, increasing columns handled per CTA without changing
weight format, activation precision, dot-product arithmetic or SwiGLU parameters.
An earlier explicit CTA4 switch was mistakenly treated as a change from CTA8;
that was a no-op and its measurements are excluded from all results below.

### Kernel Comparison

The same standalone binary compares default CTA4 against genuine CTA8, with
eight expert rows and one shared source row, rotating mode order over three
rounds with 20 warmups and 200 launches per mode. Independent binaries are not
compared. Configuration order is reversed in a second run. CUDA Graph timings
below are device-event time per invocation, excluding allocation, transfers,
preprocessing and reference checks; they are not model TPS.

| Packed Gate/Up N | K | Block | CTA4 (us) | CTA8 (us) | Latency Reduction |
|---:|---:|---:|---:|---:|---:|
| 1536 | 2048 | 64 | 13.2506 | 11.0746 | 16.42% |
| 1536 | 2048 | 128 | 13.2096 | 10.9773 | 16.90% |

Values are from the reverse-order run; the first run agrees on these shared-row
cases. CTA8 is slower for N=2048/K=768, and the eight-independent-source-row
block64 case has substantial run-to-run variation. This is not evidence for
replacing CTA4 globally or for register/occupancy causality. Eight FP16 output
files match CTA4 byte-for-byte. Two-source-row/custom-activation output checks
also match, and Compute Sanitizer memcheck reports zero errors for the tested
CTA8 variant. These are finite synthetic cases, not arbitrary-input proof.

### Full-Model Comparison

Both arms use the same mixed model, same rebuilt runtime and same GPU, with
`ORT_QMOE_INT2_PREQUANTIZED_FC1=0` and `ORT_QMOE_INT2_INT8_ACTIVATIONS=0`.
Baseline here is default FP16/INT2 CTA4, not all-INT4 or the INT8 experiment.
Each input/output configuration uses CTA4/CTA8 then CTA8/CTA4 order, two
warmups and five measured requests per process: ten requests per arm. Combined
TPS is total decode tokens divided by summed decode time, excluding prefill's
first token. The end-to-end decode definition and one-byte dense scratch guard
are retained. Profiling is disabled during timing; other GPUs on the shared
host are not controlled. The table is not a controlled head-to-head comparison
with the earlier INT8 measurements on GPU 6.

| Input Tokens | Output Tokens | CTA4 Decode TPS | CTA8 Decode TPS | Change | CTA4 Baseline Process Peak (GiB) | CTA8 Process Peak (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 111.0333 | 113.6529 | +2.36% | 26.5625 | 26.5957 |
| 128 | 128 | 110.5721 | 112.7280 | +1.95% | 26.5977 | 26.5625 |
| 512 | 128 | 108.4455 | 110.0642 | +1.49% | 26.5938 | 26.5938 |
| 2048 | 128 | 98.7603 | 99.1122 | +0.36% | 27.5938 | 27.5645 |

Process Peak uses the same `measurement_peak_process_gib` NVML measurement
definition as the fused INT8 FC1 A/B above. Each table entry is the maximum
observed peak across the forward-order and reverse-order process runs for that
configuration and arm, rounded to four decimals. It is not a pooled TPS-like
aggregation or an exact allocator/workspace peak. The mixed directions of these
small differences do not establish a consistent memory benefit or penalty from
CTA8; the longer-prompt values include process residency beyond the FC1 kernel.

Both order pairs improve TPS for 128 and 512 input tokens. For 2048 input,
the first pair is 97.8962 versus 99.4896 TPS, but the reverse pair is 99.6398
versus 98.7376 TPS (CTA4 versus CTA8). Thus the combined +0.36% does not
establish a stable long-context gain. The short-output CTA4 measurements also
show noticeable variation. No confidence interval or general speedup is claimed.
All measured token sequences match across arms within each configuration,
including all 128 generated tokens for longer-output cases; logits are finite.
This is not a full task-accuracy evaluation or proof of numerical equivalence.

A separate 128-input/16-output Nsight Systems capture confirms 360 FP16/INT2
fused FC1 CTA8 launches across 15 decode steps, with no activation quantization
path enabled. It establishes actual dispatch, not an unprofiled latency estimate.

### Dispatch Scope and Disposition

After timing, the source condition was narrowed to SM80, one source row, eight
expanded expert rows, K=2048, inter_size=768 (packed gate/up N=1536), FP16,
no bias and block64/block128. Other cases retain CTA4; existing split-K dispatch
is unchanged. The scoped GEMV unit passes formal Ninja compilation. A harness
linked against that new object matches all eight default FP16 output files, and
a separate trace shows only the two eligible shared-row cases use CTA8 while
the other six cases use CTA4. Block128 has standalone validation only; the model
measurements use block64.

The full-model timing above used the earlier broader opt-in condition, not the
subsequently scoped provider. At this documentation update, the final scoped
runtime is still rebuilding and its model smoke/dispatch check remains pending.
No completion is implied by the scoped object/harness checks. The experimental
switch stays default-off, the local ORT changes remain uncommitted and are not
shipped by this recipe. No ORT optimization PR is being opened on this evidence:
the observed short-context gain is small and the longer-context result is not
stable. Broader correctness/fallback coverage and independent repeatability
would be needed before proposing an upstream dispatch change.

Local artifacts include `qmoe-real-cta{4,8}[-reverse]-20261008.log`,
`qwen3-fp16-{cta4,cta8}[-reverse]-20261008.json`,
`qwen3-fp16-long-{cta4,cta8}[-reverse]-20261008.json` and
`qwen3-fp16-input{512,2048}-{cta4,cta8}[-reverse]-20261008.json`, with matching
logs. Actual-model dispatch is in `qwen3-cta8-dispatch-kernels-20261008.csv`;
scoped harness dispatch is in `qmoe-scoped-cta8-kernels-20261008.csv`.
`[-reverse]` denotes an optional filename suffix, not a literal path component.
These sources, binaries and raw artifacts are local, not distributed in this PR.

## Local Reproduction Artifacts

The validation machine retains `qwen3_mixed_qmoe_benchmark.py` and
`run_qwen3_block64_perf.py` under `/datadisks/disk1/jiafa/accuracy/`, and the
eight per-configuration JSON/log files, dispatch checks and `summary.json`
under `qwen3-perf-block64-20261004/`. The harnesses and raw artifacts are
local files, not distributed as part of this recipe.

The INT8-activation experiment additionally retains
`qmoe_int8_decode_experiment.cu`, `qmoe-int8-float-unpack-hash.log`,
`qmoe-int8-direct-hash.log` and `qmoe-int8-direct-a100-results.log` under
`/datadisks/disk1/jiafa/accuracy/`. Build logs and standalone binaries remain
inside `jiafa-dev` under `/tmp/`. These local prototype sources and raw artifacts
are not distributed by this recipe PR.

Follow-up artifacts include `qmoe-dp4a-before-repeat.log`,
`qmoe-dp4a-before-repeat2.log`, `qmoe-dp4a-byte-repeat.log`,
`qmoe-dp4a-byte-repeat2.log`, `qmoe-prequant-resources-before.log`,
`qmoe-dp4a-resources-byte.log`, `qmoe-dp4a-byte-memcheck.log` and
`qmoe-dp4a-byte-ncu.log` in the same local directory. The before/after binaries
are `/tmp/qmoe_prequant_experiment` and `/tmp/qmoe_dp4a_byte_experiment`
inside `jiafa-dev`; neither binary nor the prototype source is shipped here.

Full-model follow-up artifacts in the same local directory are
`qwen3-decode-mixed-current-{baseline,prequant}-20261007.json` and
`qwen3-decode-mixed-current-{baseline,prequant}-reverse-20261007.json`, with
matching logs. Dispatch evidence is retained in
`qwen3-current-prequant-dispatch-20261007.nsys-rep` and
`qwen3-current-prequant-dispatch-kernels-20261007.csv`; the complete build log
is `qmoe-prequant-runtime-build.log`. These local artifacts and the experimental
ORT source changes are not shipped in this PR. The A/B uses the Python package
under `onnxruntime/build/cuda-wheel-clean-20261002/Release`, not `wheel-site`,
with `ORT_QMOE_INT2_PREQUANTIZED_FC1` set to 0 or 1 and
`ORT_QMOE_INT2_INT8_ACTIVATIONS=0` in both arms.

Example command on that machine; repeat for both models and each length:

```bash
docker exec -w /tmp \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e PYTHONPATH=/datadisks/disk1/jiafa/accuracy/onnxruntime/build/cuda-wheel-clean-20261002/wheel-site \
  -e LD_LIBRARY_PATH=/datadisks/disk1/jiafa/cudnn9.7/lib:/datadisks/disk1/jiafa/cuda-12.8/lib64 \
  jiafa-dev /datadisks/disk1/jiafa/accuracy/onnxruntime/.venv/bin/python -u \
  /datadisks/disk1/jiafa/accuracy/qwen3_mixed_qmoe_benchmark.py \
  --physical-gpu 1 \
  --model-dir /sunghcho_data/jiafa/qwen3-30b-a3b/rtn_manual_fp16_complement_int2_int4_block64_perf_20261004 \
  --input-length 512 --output-length 128 --warmups 3 --repeats 10 \
  --output /datadisks/disk1/jiafa/accuracy/rerun-qwen3-mixed-512.json
```