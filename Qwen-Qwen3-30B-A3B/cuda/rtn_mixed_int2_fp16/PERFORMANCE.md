# Qwen3-30B-A3B Block64 Performance Analysis

Measured October 4, 2026, following the direct-ORT timing methodology in
[ONNX Runtime PR #33005](https://github.com/microsoft/onnxruntime/pull/33005).
This is a synthetic batch-one full-model comparison, not a quality benchmark
or an isolated INT2-versus-INT4 kernel experiment.

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
reference for the prototype's tilewise activation quantization is still needed.
The 2.20%-2.54% relative L2 difference is an output error, **not a task-accuracy
drop**. No model quality evaluation was run for this path.

Remaining qualification includes BF16 execution, fused SwiGLU/FC2 paths,
independent quantized-reference and broader input tests, full ORT integration,
profiling, end-to-end model decode and target-hardware runs. No H200 or Spark
results, model decode-TPS gains, memory savings or production readiness are
claimed. The prototype is local and has not been submitted as an ORT PR.

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