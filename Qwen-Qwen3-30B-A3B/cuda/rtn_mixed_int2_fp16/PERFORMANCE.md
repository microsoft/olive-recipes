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

## Local Reproduction Artifacts

The validation machine retains `qwen3_mixed_qmoe_benchmark.py` and
`run_qwen3_block64_perf.py` under `/datadisks/disk1/jiafa/accuracy/`, and the
eight per-configuration JSON/log files, dispatch checks and `summary.json`
under `qwen3-perf-block64-20261004/`. The harnesses and raw artifacts are
local files, not distributed as part of this recipe.

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