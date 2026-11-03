# Findings, methodology and limitations

This is a record of captured measurements and pinned-source analysis, not an
implementation of a production optimization. All displayed tables regenerate
with `python3 scripts/evidence.py tables`. The
[CPU verifier](scripts/evidence.py) checks table content against the committed
inputs and fails explicitly on inconsistent data.

## Evidence classes

| Class | Evidence | What it supports |
|---|---|---|
| Accepted measurement | [Request records](evidence/accepted_requests.csv) | Corrected, unprofiled runtime comparison; three workers per group |
| Instrumented diagnostic | [Allocator checkpoints](evidence/allocator_checkpoints.json) | Paired process/device NVML and both allocator domains; not headline performance |
| Experimental graph | [Logits metadata](evidence/logits_pairs.json) and pair histograms | Two-prompt numerical/token comparison and measured memory saving |
| Source-supported calculation | [QMoE layer counters](evidence/qmoe_workspace.json) | Buffer subtotal and no additional high-water growth across layers |
| Retention diagnostic | [Sequential checkpoints](evidence/sequential_requests.json) | Fifteen-request growth with stable allocator capacity; owner unresolved |
| Fresh-export check | [Fresh export record](evidence/fresh_export_mmlu.json) | One export run, artifact hashes and one MMLU sanity check against a previously saved Torch baseline; single runs, not performance or quality equivalence |

The early, superseded benchmark with incorrect timing/counting/capacity
behavior is excluded. Earlier checkpoint arena values inferred from individual
nodes are superseded by direct allocator readings. A two-call shape probe's
post-run self-check failed; it is not promoted to accepted evidence.

## Accepted benchmark

One A100-SXM4-80GB, physical GPU 0; Python 3.12.9; NVIDIA driver 580.105.08.
The measured runtime library was `libcudart.so.13`. This identifies the loaded
runtime, **not** every historical build's CUDA-toolkit version.

| Recorded dependency | Version / revision |
|---|---|
| onnxruntime-gpu | 1.30.0, `f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7` |
| onnxruntime-genai-cuda | 0.17.1, `83de55a1f3886cddbb32a5517e26adb3f9d65159` |
| llama-cpp-python | 0.3.35; separate native llama.cpp source commit not recovered |
| NumPy / psutil | 2.5.3 / 7.2.2 |
| pynvml / nvidia-ml-py | 13.0.1 / 13.610.43 |

Five labels map to measured prompts/capacities as follows:

| Context label | Actual prompt IDs | Requested sequence capacity |
|---:|---:|---:|
| 2,048 | 1,801 | 2,304 |
| 4,096 | 3,596 | 4,352 |
| 8,192 | 7,189 | 8,448 |
| 16,384 | 14,376 | 16,640 |
| 32,768 | 28,747 | 33,024 |

The synthetic repeated-text prompt retains the historical wording about a
Phi-4 runtime comparison. It was used identically for this Qwen experiment;
it is not a chat-template or representative quality-evaluation workload.

Each context/variant/repetition uses a fresh worker. Its first request is
`cold`; `warm_1` and `warm_2` retain that worker's model but create fresh
request state, with no prefix reuse. All 135 input hashes and actual
completion counts/stops are checked; no warmups or failures are discarded.

Inference TTFT starts before full prompt evaluation (`append_tokens()` on
ORT) and ends at the first CPU-available ID. It excludes model loading,
tokenization and fresh request setup; `request_ttft_ms` additionally includes
setup. Decode throughput is `(completion_IDs - 1) / (last_time - first_time)`,
including an observed terminal EOS ID when present. llama.cpp counts IDs,
not streamed text chunks.

Capacities are matched and verified. A separate CPU process samples NVML
every nominal 10 ms; acquisition start and end must fit inside a selected
phase. Cleanup and post-timing tensor inspection are excluded from request
peaks. `peak_vram_mib` combines shared model preparation with that request's
setup/inference samples; `request_peak_vram_mib` excludes shared loading.
The tables use the former consistently. Both are retained in the CSV.

Cells below are median `[min, max]` across three independent repetitions.
They are observed ranges, not confidence intervals. Nominal labels are not
actual token counts. Cold and warm results are never pooled.

<!-- BEGIN accepted -->
### cold

| Prompt tokens (label) | Variant | Outputs / stop | Peak device MiB | TTFT ms | Decode IDs/s |
|---:|---|---|---:|---:|---:|
| 1,801 (2,048) | ORT baseline | 64/length | 33,488.88 [33,486.88, 33,492.88] | 8,504.07 [8,484.87, 8,513.37] | 79.22 [76.88, 79.31] |
| 1,801 (2,048) | ORT Reserve | 64/length | 18,950.88 [18,950.88, 18,950.88] | 8,692.39 [8,656.84, 8,717.70] | 75.61 [75.57, 76.16] |
| 1,801 (2,048) | llama.cpp | 55/eos | 18,596.88 [18,596.88, 18,596.88] | 668.36 [668.16, 668.66] | 143.90 [143.89, 143.97] |
| 3,596 (4,096) | ORT baseline | 64/length | 34,772.88 [34,762.88, 35,790.88] | 15,359.56 [15,339.32, 15,405.76] | 74.71 [72.79, 74.74] |
| 3,596 (4,096) | ORT Reserve | 64/length | 20,870.88 [20,870.88, 20,870.88] | 15,602.21 [15,526.31, 15,612.42] | 74.93 [74.85, 74.97] |
| 3,596 (4,096) | llama.cpp | 64/length | 18,858.88 [18,858.88, 18,858.88] | 1,158.18 [1,155.45, 1,160.75] | 131.77 [131.71, 132.04] |
| 7,189 (8,192) | ORT baseline | 64/length | 39,388.88 [39,380.88, 40,588.88] | 29,467.91 [29,424.35, 29,489.94] | 72.53 [72.42, 73.61] |
| 7,189 (8,192) | ORT Reserve | 64/length | 25,222.88 [25,222.88, 25,222.88] | 29,681.10 [29,674.10, 29,690.91] | 72.44 [72.40, 72.53] |
| 7,189 (8,192) | llama.cpp | 64/length | 19,618.88 [19,618.88, 19,618.88] | 2,451.80 [2,412.41, 2,455.38] | 114.66 [114.46, 115.25] |
| 14,376 (16,384) | ORT baseline | 64/length | 46,540.88 [46,538.88, 46,542.88] | 31,257.10 [31,247.24, 31,388.65] | 70.77 [70.43, 70.83] |
| 14,376 (16,384) | ORT Reserve | 64/length | 32,902.88 [32,902.88, 32,902.88] | 31,559.98 [31,556.42, 31,562.41] | 70.69 [70.54, 71.04] |
| 14,376 (16,384) | llama.cpp | 64/length | 21,138.88 [21,138.88, 21,138.88] | 6,042.23 [6,035.29, 6,048.47] | 90.23 [89.97, 90.38] |
| 28,747 (32,768) | ORT baseline | 64/length | 60,882.88 [60,874.88, 60,892.88] | 39,599.56 [39,568.81, 39,673.71] | 66.78 [66.65, 67.52] |
| 28,747 (32,768) | ORT Reserve | 64/length | 47,238.88 [47,238.88, 47,238.88] | 39,796.20 [39,764.64, 39,820.80] | 66.99 [66.89, 67.15] |
| 28,747 (32,768) | llama.cpp | 64/length | 24,178.88 [24,178.88, 24,178.88] | 18,475.03 [18,471.36, 18,486.57] | 60.99 [60.88, 61.03] |

### warm_1

| Prompt tokens (label) | Variant | Outputs / stop | Peak device MiB | TTFT ms | Decode IDs/s |
|---:|---|---|---:|---:|---:|
| 1,801 (2,048) | ORT baseline | 64/length | 33,494.88 [33,492.88, 33,498.88] | 173.22 [173.09, 173.52] | 168.08 [166.51, 168.90] |
| 1,801 (2,048) | ORT Reserve | 64/length | 18,956.88 [18,956.88, 18,956.88] | 173.13 [173.03, 173.61] | 168.91 [168.48, 169.04] |
| 1,801 (2,048) | llama.cpp | 55/eos | 18,596.88 [18,596.88, 18,596.88] | 459.37 [459.32, 459.87] | 148.53 [148.45, 148.65] |
| 3,596 (4,096) | ORT baseline | 64/length | 34,778.88 [34,768.88, 35,796.88] | 364.56 [364.30, 364.72] | 163.26 [162.41, 163.47] |
| 3,596 (4,096) | ORT Reserve | 64/length | 20,876.88 [20,876.88, 20,876.88] | 363.28 [362.72, 363.78] | 162.93 [162.84, 163.02] |
| 3,596 (4,096) | llama.cpp | 64/length | 18,858.88 [18,858.88, 18,858.88] | 962.20 [960.09, 963.44] | 134.69 [134.53, 134.95] |
| 7,189 (8,192) | ORT baseline | 64/length | 39,394.88 [39,386.88, 40,594.88] | 930.77 [894.28, 941.52] | 151.50 [151.27, 151.90] |
| 7,189 (8,192) | ORT Reserve | 64/length | 25,228.88 [25,228.88, 25,228.88] | 928.11 [910.41, 932.17] | 150.99 [150.08, 151.30] |
| 7,189 (8,192) | llama.cpp | 64/length | 19,618.88 [19,618.88, 19,618.88] | 2,207.08 [2,205.49, 2,208.08] | 116.97 [116.88, 117.73] |
| 14,376 (16,384) | ORT baseline | 64/length | 46,546.88 [46,544.88, 46,548.88] | 2,676.28 [2,661.26, 2,677.18] | 143.51 [143.24, 143.58] |
| 14,376 (16,384) | ORT Reserve | 64/length | 32,908.88 [32,908.88, 32,908.88] | 2,672.73 [2,672.23, 2,712.47] | 143.54 [143.34, 143.84] |
| 14,376 (16,384) | llama.cpp | 64/length | 21,138.88 [21,138.88, 21,138.88] | 5,830.23 [5,830.03, 5,838.77] | 91.60 [91.58, 91.72] |
| 28,747 (32,768) | ORT baseline | 64/length | 60,888.88 [60,880.88, 60,898.88] | 10,631.84 [10,588.51, 10,644.02] | 127.20 [127.06, 128.33] |
| 28,747 (32,768) | ORT Reserve | 64/length | 47,244.88 [47,244.88, 47,244.88] | 10,641.62 [10,612.20, 10,660.30] | 127.75 [127.68, 127.85] |
| 28,747 (32,768) | llama.cpp | 64/length | 24,178.88 [24,178.88, 24,178.88] | 18,269.86 [18,262.39, 18,276.91] | 61.57 [61.45, 61.72] |

### warm_2

| Prompt tokens (label) | Variant | Outputs / stop | Peak device MiB | TTFT ms | Decode IDs/s |
|---:|---|---|---:|---:|---:|
| 1,801 (2,048) | ORT baseline | 64/length | 33,500.88 [33,498.88, 33,504.88] | 173.28 [173.21, 173.37] | 169.00 [168.62, 169.54] |
| 1,801 (2,048) | ORT Reserve | 64/length | 18,962.88 [18,962.88, 18,962.88] | 172.94 [172.85, 173.10] | 168.82 [168.65, 168.92] |
| 1,801 (2,048) | llama.cpp | 55/eos | 18,596.88 [18,596.88, 18,596.88] | 459.70 [459.32, 459.71] | 149.32 [149.12, 149.34] |
| 3,596 (4,096) | ORT baseline | 64/length | 34,784.88 [34,774.88, 35,802.88] | 364.50 [364.36, 364.52] | 163.31 [163.25, 163.56] |
| 3,596 (4,096) | ORT Reserve | 64/length | 20,882.88 [20,882.88, 20,882.88] | 363.39 [362.97, 363.59] | 163.31 [162.02, 163.69] |
| 3,596 (4,096) | llama.cpp | 64/length | 18,858.88 [18,858.88, 18,858.88] | 960.88 [960.41, 961.85] | 134.67 [134.61, 134.90] |
| 7,189 (8,192) | ORT baseline | 64/length | 39,400.88 [39,392.88, 40,600.88] | 920.05 [894.48, 932.89] | 151.49 [151.36, 152.02] |
| 7,189 (8,192) | ORT Reserve | 64/length | 25,234.88 [25,234.88, 25,234.88] | 931.34 [917.70, 932.97] | 150.83 [150.57, 151.92] |
| 7,189 (8,192) | llama.cpp | 64/length | 19,618.88 [19,618.88, 19,618.88] | 2,206.20 [2,205.96, 2,206.27] | 117.52 [117.21, 117.84] |
| 14,376 (16,384) | ORT baseline | 64/length | 46,552.88 [46,550.88, 46,554.88] | 2,678.36 [2,651.11, 2,681.34] | 143.75 [143.50, 143.83] |
| 14,376 (16,384) | ORT Reserve | 64/length | 32,914.88 [32,914.88, 32,914.88] | 2,673.56 [2,649.52, 2,705.87] | 143.27 [143.26, 143.70] |
| 14,376 (16,384) | llama.cpp | 64/length | 21,138.88 [21,138.88, 21,138.88] | 5,831.51 [5,827.29, 5,838.75] | 91.67 [90.89, 91.94] |
| 28,747 (32,768) | ORT baseline | 64/length | 60,894.88 [60,886.88, 60,904.88] | 10,634.83 [10,595.34, 10,647.48] | 127.69 [127.42, 128.20] |
| 28,747 (32,768) | ORT Reserve | 64/length | 47,250.88 [47,250.88, 47,250.88] | 10,648.76 [10,605.92, 10,661.31] | 128.23 [127.91, 128.41] |
| 28,747 (32,768) | llama.cpp | 64/length | 24,178.88 [24,178.88, 24,178.88] | 18,272.06 [18,263.57, 18,277.02] | 61.58 [61.57, 61.64] |
<!-- END accepted -->

## Exact memory definitions

MiB is bytes / 2^20; GiB is bytes / 2^30. Earlier rounded values such as
47.24 and 30.85 were thousands of MiB, not GiB: the corresponding
device-used peaks are approximately 46.13 and 30.13 GiB.

The accepted benchmark recorded NVML **device-used v2** on an otherwise idle
GPU. The later direct-counter diagnostic recorded both device and process
counters. Device-used exceeded process-used by exactly 8.875 MiB in that
diagnostic. The separately recorded driver-reserved counter, 767.25 MiB,
is not added to process memory.

For each allocator:

```text
live allocation blocks = InUse
logical requested bytes = RequestedInUse
padding attached to live blocks = InUse - RequestedInUse
unused retained capacity = TotalAllocated - InUse
total held device allocation = TotalAllocated
lifetime high-water = MaxInUse (audit only; not an additive current component)
```

[BFC Reserve](https://github.com/microsoft/onnxruntime/blob/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7/onnxruntime/core/framework/bfc_arena.cc#L275-L292)
includes device-allocator initializer allocations in `InUse` and
`TotalAllocated`. Therefore these counters are not exclusively an
"activation arena". Arena slack is not automatically a leak.

## Experimental allocator decomposition

The original lifecycle capture lacked direct GenAI counters. Two later
fresh-process original-artifact requests captured the real allocator instances
through a read-only, exact-header native adapter and `AllocatorGetStats`.
No runtime rebuild, allocator change, graph change or full-matrix rerun was
performed for that reconciliation.

Fixed checkpoints are device-synchronized. Joint peak samples bracket NVML
with allocator reads and exclude unequal endpoint counters. Equal endpoints
do not exclude a short-lived change within the read interval. D/F use one actual
stable tuple, **not** separate maxima added together. GenAI KV/logits shapes
at D/F use source-verified phase lifetimes and direct E/G metadata; mutable
generator I/O was not traversed concurrently.

| Point | Meaning |
|---|---|
| A | Before CUDA/model initialization |
| B | Model/session loaded, before Generator |
| C | Generator ready, before generation |
| D | Stable sampled prefill peak |
| E | First token produced |
| F | Stable sampled decode peak |
| G | Generation complete, request objects alive |
| H | Generator/params destroyed, model still alive |

Both runs reproduced the previously captured device-used lifecycle values
and all accepted output IDs. Median joint gaps were about 10.8 ms; maxima
were 80.1 / 311.5 ms. These are sampled peaks, not allocation-event absolute
maxima. All 24 checkpoints, including CUDA-only and teardown controls,
are in [the compact input](evidence/allocator_checkpoints.json).

The following current-checkpoint tables are in MiB. Accounted totals include
the measured 416 MiB CUDA-initialization process baseline. "GenAI other +
live padding" combines two separately reconstructible fields for readability.

### 7,189 tokens

<!-- BEGIN allocator-7k -->
| Point | Main live | Main unused | KV | Logits | GenAI other + live padding | GenAI unused | Accounted incl. CUDA baseline | NVML process | Residual (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 (n/a) |
| B | 15,976.95 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 16,393.90 | 16,462.00 | 68.10 (0.4137%) |
| C | 15,976.95 | 0.95 | 792.00 | 0.00 | 33.53 | 199.47 | 17,418.90 | 17,488.00 | 69.10 (0.3951%) |
| D | 16,729.32 | 2,832.59 | 792.00 | 2,083.34 | 33.71 | 2,211.96 | 25,098.90 | 25,176.00 | 77.10 (0.3062%) |
| E | 15,976.95 | 3,584.95 | 792.00 | 2,083.34 | 39.81 | 2,205.86 | 25,098.90 | 25,176.00 | 77.10 (0.3062%) |
| F | 15,976.95 | 3,584.95 | 792.00 | 0.29 | 39.94 | 4,288.77 | 25,098.90 | 25,214.00 | 115.10 (0.4565%) |
| G | 15,976.95 | 3,584.95 | 792.00 | 0.29 | 39.94 | 4,288.77 | 25,098.90 | 25,214.00 | 115.10 (0.4565%) |
| H | 15,976.95 | 3,584.95 | 0.00 | 0.00 | 0.00 | 5,121.00 | 25,098.90 | 25,214.00 | 115.10 (0.4565%) |
<!-- END allocator-7k -->

### 28,747 tokens

<!-- BEGIN allocator-28k -->
| Point | Main live | Main unused | KV | Logits | GenAI other + live padding | GenAI unused | Accounted incl. CUDA baseline | NVML process | Residual (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 (n/a) |
| B | 15,976.95 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 16,393.90 | 16,462.00 | 68.10 (0.4137%) |
| C | 15,976.95 | 0.95 | 3,096.00 | 0.00 | 201.88 | 799.12 | 20,490.90 | 20,560.00 | 69.10 (0.3361%) |
| D | 21,958.29 | 4,259.61 | 3,096.00 | 8,330.73 | 202.75 | 8,851.52 | 47,114.90 | 47,192.00 | 77.10 (0.1634%) |
| E | 15,976.95 | 10,240.95 | 3,096.00 | 8,330.73 | 208.77 | 8,845.50 | 47,114.90 | 47,192.00 | 77.10 (0.1634%) |
| F | 15,976.95 | 10,240.95 | 3,096.00 | 0.29 | 208.59 | 17,176.12 | 47,114.90 | 47,230.00 | 115.10 (0.2437%) |
| G | 15,976.95 | 10,240.95 | 3,096.00 | 0.29 | 208.59 | 17,176.12 | 47,114.90 | 47,230.00 | 115.10 (0.2437%) |
| H | 15,976.95 | 10,240.95 | 0.00 | 0.00 | 0.00 | 20,481.00 | 47,114.90 | 47,230.00 | 115.10 (0.2437%) |
<!-- END allocator-28k -->

**Verdict: bounded but not uniquely attributable.** At G, coverage is
99.54% / 99.76%, with residual 115.097649 MiB in both single-request cases.
Teardown controls split that footprint into 56.097649 MiB associated with
model lifetime, 11 MiB with GenAI-global lifetime, and 48 MiB surviving
GenAI shutdown beyond the initial CUDA baseline. These are measured lifetime
buckets, not proof of particular cuBLAS, CUDA module or driver owners.

Absent allocators are represented explicitly as null. The CUDA-only and
post-GenAI-shutdown controls do not invoke allocator creation to obtain
statistics; zero allocator terms there follow absence/source-verified
teardown, while NVML remains a directly observed counter.

After H, GenAI live/requested bytes are zero but 5,121 / 20,481 MiB remains
reserved. Model destruction alone does not release this global arena;
GenAI shutdown does. Full process exit returned the GPU to idle.
This is retained capacity, not a demonstrated cross-process leak.

## Tensor payload and allocator ownership

The graph's 828 initializers sum to 16,752,996,536 bytes
(15,976.90 MiB). [Category totals](evidence/provenance.json) include packed
INT4 attention/expert weights, FP16 scales, embeddings/head/router/norms,
rotary caches and constants. UINT8 storage contains packed 4-bit values.
Graph storage bytes are not an independent GPU-residency measurement:
the CUDA model/session counters provide the separate runtime observation.

<!-- BEGIN weights -->
| Initializer storage category | MiB |
|---|---:|
| expert_int4_packed_bytes | 13,824.000000 |
| expert_scales_fp16 | 432.000000 |
| attention_int4_packed_bytes | 432.000000 |
| attention_scales_fp16 | 13.500000 |
| embeddings_fp16 | 593.500000 |
| lm_head_fp16 | 593.500000 |
| router_fp16 | 24.000000 |
| norms_fp16 | 0.402344 |
| rotary_cache_fp16 | 64.000000 |
| graph_constants_int64 | 0.000175 |
| **Total** | **15,976.902519** |
<!-- END weights -->

Packed expert storage is `layers * experts * 3 * hidden * intermediate / 2`.
Scale storage uses the actual group size 128. Attention projection widths
are `32 * 128` for Q and `4 * 128` for K/V, with hidden width 2048.
The verifier independently reconstructs category bytes from the committed
architecture facts; graph constants retain their observed 184-byte subtotal.

KV payload is exactly:

```text
48 layers * 2(K,V) * 4 KV heads * sequence_capacity * 128 head_dim * 2 FP16 bytes
capacity 8448  -> 830472192 bytes  -> 792 MiB
capacity 33024 -> 3246391296 bytes -> 3096 MiB
```

The native snapshot observed 96 unique tensors with those FP16 shapes;
shared past/present storage is counted once.
[State construction](https://github.com/microsoft/onnxruntime-genai/blob/83de55a1f3886cddbb32a5517e26adb3f9d65159/src/models/decoder_only.cpp#L25-L35)
and the
[KV constructor](https://github.com/microsoft/onnxruntime-genai/blob/83de55a1f3886cddbb32a5517e26adb3f9d65159/src/models/io/static_kv_cache.cpp#L344-L382)
allocate KV before the first Run.

Raw FP16 logits are `[1, actual_prompt_tokens, 151936]`:

```text
7189  * 151936 * 2 = 2184535808 bytes = 2083.335693 MiB
28747 * 151936 * 2 = 8735408384 bytes = 8330.734619 MiB
one decode row     = 303872 bytes    = 0.289795 MiB
```

[Logits::Update](https://github.com/microsoft/onnxruntime-genai/blob/83de55a1f3886cddbb32a5517e26adb3f9d65159/src/models/io/logits.cpp#L84-L114)
allocates this output before session Run. GenAI's
[global device allocator](https://github.com/microsoft/onnxruntime-genai/blob/83de55a1f3886cddbb32a5517e26adb3f9d65159/src/models/model.cpp#L310-L368)
owns KV/logits/other device buffers. The
[per-node profiler](https://github.com/microsoft/onnxruntime/blob/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7/onnxruntime/core/framework/sequential_executor.cc#L395-L405)
samples the model-session allocator, not process-wide CUDA memory.

## QMoE workspace: source subtotal and layer reuse

The first QMoE node increases high-water usage and held capacity, with zero
net live allocation after Compute returns. Later layers reuse that capacity.
The committed evidence retains all 48 layer-local before/after counters;
two scalar-arena events were excluded from differences between GPU-arena
events, avoiding spurious approximately 16 GiB cross-allocator jumps.

The source-based subtotal includes overlapped input/output buffers, a
dedicated FC1 result, split-K2 SwiGLU partials, routing/permutation buffers
and pointer arrays. It scales mainly with `prompt_tokens * top_k` and the
hidden/intermediate widths. The subtotal omits small aligned count metadata:
it is not an exact formula for arbitrary routes or hardware.

<!-- BEGIN qmoe -->
| Case | Calculated lower bound MiB | Observed QMoE peak jump MiB | Unmodeled small metadata MiB | Reservation growth MiB | Layers 1-47 extra reservation |
|---|---:|---:|---:|---:|---:|
| 7k | 1212.14 | 1212.15 | 0.008263 | 3072.00 | 0 bytes |
| 28k | 4847.02 | 4847.05 | 0.028587 | 8192.00 | 0 bytes |
<!-- END qmoe -->

[QMoE scratch allocation](https://github.com/microsoft/onnxruntime/blob/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7/onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc#L1128-L1144)
uses ORT's allocator. The
[workspace layout](https://github.com/microsoft/onnxruntime/blob/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7/onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_kernels.cu#L2129-L2273)
is passed into the runner; CUTLASS scratch must not be added again outside
model-session `TotalAllocated`. Cross-layer accumulation is disproved;
buffer necessity/optimality is not.

Prefill capture is disabled by GenAI's
[capture-length gate](https://github.com/microsoft/onnxruntime-genai/blob/83de55a1f3886cddbb32a5517e26adb3f9d65159/src/models/decoder_only.cpp#L74-L90)
with a default ceiling of one token. First-use latency must not be
explained as prefill CUDA graph replay. QMoE's
[tactic cache uses M buckets](https://github.com/microsoft/onnxruntime/blob/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7/onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.cc#L145-L182),
not necessarily one entry per exact token count. This PR does not establish
a universal cold-start improvement.

## Experimental last-row logits candidate

The isolated graph changes only the LM-head input path:

```text
final normalized hidden states [B,S,2048]
  -> Gather(axis=1, scalar index=-1) [B,2048]
  -> Unsqueeze(axis=1) [B,1,2048]
  -> lm_head MatMul [B,1,151936]
```

The original graph, configuration and external data hashes stayed unchanged.
This mirrors the existing
[GenAI builder pattern](https://github.com/microsoft/onnxruntime-genai/blob/83de55a1f3886cddbb32a5517e26adb3f9d65159/src/python/py/models/builders/base.py#L5893-L5912);
it is a graph rewrite, not a runtime flag for the original Mobius artifact.
Neither the candidate graph nor weights are committed.

The direct GenAI held-capacity increase C -> E is 4,096 / 16,384 MiB.
Independent original/candidate sampled device-peak differences match those
amounts exactly. This does not assign every free chunk permanently to logits;
the allocator is shared.

<!-- BEGIN logits -->
| Case | Original MiB | Candidate MiB | Saved MiB | Max absolute diff | Mean absolute diff | Max ordered FP16 steps | Differing FP16 values | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 7k | 25,222.88 | 21,126.88 | 4,096.00 | 0.015625 | 0.0005331974 | 188 | 22,160 | 64/64 match |
| 28k | 47,238.88 | 30,854.88 | 16,384.00 | 0.015625 | 0.0005309108 | 580 | 22,670 | 64/64 match |
<!-- END logits -->

**Not exact-preserving; not accepted as production-ready.** Two prompts with
matching 64 greedy IDs are not broad correctness or output-quality validation.
Original/original and candidate/candidate repeatability controls, followed
by an identical-hidden-state/weight LM-head isolation, have not been run.
Kernel selection is a hypothesis, not an established numerical root cause.

The metric counts adjacent finite binary16 values in monotonic order, merging
signed zeros for numeric distance. Maximum distance occurs near zero; the
largest absolute difference occurs at larger logits and can be one local
ULP. Do not equate 188/580 with Phi-4's reported four-step result without
verifying that earlier metric definition.

The compact evidence stores every unequal `(original bits, candidate bits,
count)` pair plus the equal-pair count. It reproduces max/mean absolute and
ordered-step differences exactly, but does not retain vocabulary order or
complete logits vectors. Raw-vector SHA-256 fingerprints are included.

Likewise, removing machine identifiers preserves numeric recalculation but
does not let a CPU-only reviewer independently re-observe worker process
isolation, GPU residency, timing boundaries or actual allocator calls.
Those were checked against raw records locally before projection. The compact
evidence supports reproducible calculations and documented captured results,
not an independently executed hardware measurement.

**Exact-byte correction:** raw profiles show candidate model-session held
capacity is **8 bytes larger** at both contexts, including at the first node.
The added graph has two INT64 constants; their individual GPU placement was
not traced. The older two-decimal-MiB tables hid this distinction. QMoE
workspace deltas are unchanged and the measured 4/16 GiB saving stands.

## Fifteen sequential identical requests

One original model/process/session, new Generator/params each time, 7,189
input IDs, capacity 8,448, greedy 64-ID cap, profiling disabled and CUDA graph
settings unchanged. Five synchronized checkpoints per request capture both
allocator domains and process/device NVML. All 960 IDs match the accepted
baseline; full worker exit released GPU memory.

<!-- BEGIN sequential -->
| Request | Post-cleanup process MiB | Increase MiB | Model live MiB | Model held MiB | GenAI held MiB | GenAI live/requested bytes | Tokens |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 25,214.00 | 0.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 2 | 25,220.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 3 | 25,226.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 4 | 25,232.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 5 | 25,238.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 6 | 25,244.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 7 | 25,250.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 8 | 25,256.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 9 | 25,262.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 10 | 25,270.00 | 8.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 11 | 25,276.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 12 | 25,282.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 13 | 25,288.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 14 | 25,294.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
| 15 | 25,300.00 | 6.00 | 15,976.95 | 19,561.90 | 5,121.00 | 0/0 | 64/64 match |
<!-- END sequential -->

Post-cleanup process usage increases by **86 MiB**. The request-2-to-15
OLS slope is **6.210989 MiB/request**; no plateau appears in fifteen requests.
Model live/requested/held bytes are constant after the first request.
GenAI held bytes are constant and live/requested bytes return to zero each
time. The additional footprint is outside these recorded counters.

The owner remains unresolved. Same token shape is not necessarily the same
CUDA graph annotation ID, but no graph-disable/varying-shape control or
allocation-event trace was run. The observation does not prove an
indefinite leak, nor that graphs alone own the increase. The single-request
115.10 MiB residual must not be described as universally constant overhead.

## Fresh export and MMLU sanity check

One fresh run of the documented recipe was made on 2026-10-08, after this PR was
opened. It does not replace the archived artifact or any accepted result above.
Raw logs and per-question samples are not committed. The compact record is
[fresh_export_mmlu.json](evidence/fresh_export_mmlu.json); `python3
scripts/evidence.py verify --check-docs` recomputes the accuracies, deltas, interval
and exact p-values from its paired counts, checks the artifact size/hash relations
against the provenance record, and fails if the tables below differ.

The pinned Olive commit imports `requests` without declaring it (the same symptom
is reported upstream for Olive 0.13.0 in
[Olive #2676](https://github.com/microsoft/Olive/issues/2676), with a fix proposed
in [#2693](https://github.com/microsoft/Olive/pull/2693)), so the documented
`olive run` stopped at import until `requests` was installed. It is now listed in
`cuda/requirements.txt`. The Hugging Face snapshot was already in the local cache,
so download time is not included in the export timing.

<!-- BEGIN fresh-export -->
| Step | Result |
|---|---|
| `pip install -r cuda/requirements.txt` in a clean environment | exit 0, 168 s |
| `olive run --config cuda/kquant_fp16/config.json` as documented | exit 1 after 0 s: ModuleNotFoundError: No module named 'requests' (olive/telemetry/library/exporter.py, line 14) |
| The same command after adding `requests==2.34.2` and its transitive dependencies | exit 0, 371 s (KQuant pass 286.1 s, MobiusBuilder pass 50.7 s); peak GPU 0 memory 13,581 MiB |
| ORT GenAI smoke test on the fresh export | `391`; the 7,189-token greedy run matched 64/64 accepted IDs |

| File | Fresh bytes | Archived bytes | SHA-256 |
|---|---:|---:|---|
| `model.onnx.data` | 16,753,033,216 | 16,753,033,216 | identical (`7068abc4...c34f`) |
| `model.onnx` | 903,664 | 903,556 | differs only in `graph.name` (219 versus 112 characters: +107 characters and +1 length-prefix byte = +108 bytes) |
| `genai_config.json` | 1,665 | 1,341 | differs in raw bytes; JSON content identical |

The op-type histogram, node count (1,163), initializer count (828), initializer names and input/output name counts (98 / 97) are identical.
<!-- END fresh-export -->

This is a single run with the pinned sources; it does not identify the historical
Olive/Mobius versions.

**MMLU.** Olive's lm-eval evaluator compared the fresh export with a previously
saved Torch bf16 run of the same checkpoint. Torch was not rerun in this check.

<!-- BEGIN mmlu -->
| Side | Correct / 9,183 | Accuracy (stderr, points) |
|---|---:|---:|
| ONNX fresh export | 7,551 | 82.23% (0.40) |
| Torch bf16 (previously saved baseline, not rerun) | 7,649 | 83.30% (0.39) |
| ONNX archived artifact | 7,551 | 82.23% (0.40) |

| Paired comparison | Both right | Only first right | Only second right | Both wrong | Delta (points) | 95% CI (points) | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX fresh minus Torch bf16 | 7,381 | 170 | 268 | 1,364 | -1.07 | -1.51 to -0.62 | 3.3e-06 |
| ONNX fresh versus archived ONNX | 7,551 | 0 | 0 | 1,632 | +0.00 | 0.00 to 0.00 | 1 |

- Protocol: Olive LMEvaluator (ortgenai) with lm-eval 0.4.13 task mmlu; test split; limit 200 per subject; 0-shot; no chat template; batch size 1; max_length 4096.
- CI method: Normal approximation: delta +/- 1.96 * SE; SE = sample standard deviation (ddof=1) of the per-question paired differences divided by sqrt(n); the deviation is reconstructed exactly from the paired counts. No resampling, no seed.
- Exact McNemar: Two-sided exact binomial test (p = 0.5) on the discordant pairs.
- Captured, not recomputable from the compact evidence: both sides chose the same option for 94.23% of questions, and 60 of 9,183 prompts exceed 512 tokens (reconstructed from the standard 0-shot layout, approximately +/-20 tokens).
<!-- END mmlu -->

Limits: this is a sanity check of the exported weights, not quality equivalence.
Olive's `ortgenai` backend appears to replace the exported `genai_config.json`
provider options with defaults (read from `olive/evaluator/lmeval_ort.py`, not
confirmed at run time), so the shipped CUDA-graph and strict skip-layer-norm
settings may not have been in effect. Almost no prompt exceeds 512 tokens, so the
run says nothing about chunked prefill or long-context behavior. Every number is a
single run.

## Outstanding work

Historical configuration-limited comparison: the experimental ORT-candidate minus
llama.cpp deltas are 1,508 MiB (1.47 GiB) at 7K and 6,676 MiB (6.52 GiB) at 28K, a
difference of 5,168 MiB (5.05 GiB). The experimental candidate prefilled the full
prompt in one pass (n=1, instrumented, numerically unaccepted), whereas llama.cpp
processed it in 512-token micro-batches (llama-cpp-python 0.3.35 defaults
`n_batch = n_ubatch = 512`, flash attention off). The QMoE workspace measurements
above show that prompt-sized allocations depend on the number of tokens processed
concurrently, so these deltas describe those runs but do not establish an intrinsic
ORT-versus-llama.cpp runtime gap. Matched-prefill testing was performed separately;
its results are outside this PR's evidence, which neither includes nor relies on
them. No fresh paired candidate/llama.cpp medians are claimed here.

The accepted benchmark compares the tested default prefill configurations: ORT's
whole-prompt prefill (default allocator and the opt-in initializer-Reserve setting)
and llama.cpp's default 512-token micro-batches. In it, both ORT variants had higher
sampled peaks than llama.cpp at every context tier and request kind. This PR makes
no tuned-parity, quality-equivalence or universal runtime-parity claim.

Next: any claim based on the separate matched-prefill testing needs its own
committed, verifier-checked evidence; explain numerical differences before choosing
a tolerance; attribute repeated-request growth separately. Arena disabling and
allocation/free stack capture are possible diagnostic controls, not production
recommendations. Coordinate Mobius/logits, workspace/resource-accounting and
expert-offloading owners instead of creating duplicate implementations.
