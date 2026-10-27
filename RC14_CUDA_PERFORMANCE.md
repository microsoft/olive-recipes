# RC14 INT3 Model: ONNX Runtime CUDA Performance

Measured on 2026-10-27. This is a **single-runtime, single-GPU measurement**
of the supplied `mai_coder_bf16_quant_int3` artifact, not a comparison with
another runtime, implementation, or model.

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
| `mai_coder_bf16_quant_int3.onnx` | 630457 | `91dc2b0031aae23bea17c7e1be9c7713cf290b7c931f0a60972bb1d4bd440a5c` |
| `mai_coder_bf16_quant_int3.data` | 70574209280 | `6bcdb247f1b01574e3e9fdd97fe0d989f658d8303ebce7b41f15844093c51937` |

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

## All Measured Request Timings

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

## Scope and Retained Artifacts

The raw per-process JSON includes unrounded timings, all predicted token
streams, logits hashes, prompt hashes, mapped binary hashes, GPU identity, and
sampling interval statistics. Model hashes, environment records, worker logs,
ORT profiles, and the Nsight capture/CSV are retained locally under
`rc14-cuda-results-20261027`; they are not added to this PR. The local harness
and raw captures are not a public self-contained reproduction package.

All four input lengths completed without OOM or non-finite last-position
logits. Two rounds and ten measured requests per length are a limited
controlled sample, not confidence intervals or production variability
coverage. No throughput ratios, memory savings against another runtime,
cross-runtime numerical differences, or task-accuracy claims are made.
