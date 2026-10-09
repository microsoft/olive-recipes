# Qwen3-30B-A3B llama.cpp GGUF Performance

Measured October 9, 2026. This document follows the results-table layout of
[the October 8 ORT performance report](https://github.com/microsoft/olive-recipes/blob/e67424fb95b7c435c481608a78f58cc9095a7871/Qwen-Qwen3-30B-A3B/cuda/rtn_mixed_int2_fp16/PERFORMANCE.md)
and compares llama.cpp Q2_K and Q4_K_M models. The TTFT and context-dependent
decode columns below are new request-level measurements, not estimates from
`llama-bench` prompt-processing or prompt-plus-generation throughput.

## Environment and Models

- NVIDIA A100-SXM4-80GB, physical GPU 6, SM80; Docker `jiafa-dev`.
- GPU UUID: `GPU-11437cd6-cf80-c4ff-cd96-31b15c9ee949`.
- llama.cpp source: `c479922ac520a08969b4c1dc154d7bbb3c386d85`.
  No tracked source changes were present; unrelated untracked files existed.
- Separate source-matching Release build, CUDA 12.8, CUDA architecture 80:
  `build-qwen3-compare-20261009`. The pre-existing binary rejected `qwen3moe`
  and was not used. Binary build metadata may be unavailable because of Git
  ownership restrictions during configuration; the source revision is recorded
  independently.
- 49/49 layers offloaded to one GPU, split mode none; FP16 K/V cache;
  Flash Attention enabled; logical batch 2048, microbatch 512; 8 CPU threads
  for both prompt processing and generation. Batch size is one sequence.
- Default CUDA Graph behavior retained. Warmup logs confirm CUDA Graph reuse.
  Runtime INFO/DEBUG logs are disabled before measurement; warnings/errors
  remain enabled. No profiler is active during measurement.
- GGUF repository:
  [`unsloth/Qwen3-30B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/tree/d5b1d57bd0b504ac62ae6c725904e96ef228dc74),
  pinned revision `d5b1d57bd0b504ac62ae6c725904e96ef228dc74`.
- Both files contain 30,532,122,624 stored model parameters: 48 MoE layers,
  128 experts per layer, top-k 8. This parameter count includes inactive experts;
  it is not the approximately 3B active-parameter count per token.

## Measurement Method

Input lengths 128/512/2048/4096; exactly 128 output tokens; three warmups and
ten measured requests per model/length. Each of the eight configurations runs
in a fresh process on the same GPU. Order is Q4_K_M/Q2_K at 128, Q2_K/Q4_K_M
at 512, Q4_K_M/Q2_K at 2048, and Q2_K/Q4_K_M at 4096. This is alternating
configuration order, not repeated interleaved A/B trials for each length.

A standalone C++ harness calls the matching llama.cpp C API directly. It uses
the same repeated Paris prompt, Qwen chat delimiters, `/no_think` suffix and
empty completed thinking prefix as [PERFORMANCE.md](PERFORMANCE.md). Prompt
token IDs were checked against the ORT harness's tokenizer at every length;
they match exactly for both GGUF files. Tokenization occurs before timing.
Greedy CPU argmax ignores EOS to enforce exactly 128 output tokens. All logits
used for selection are checked for finiteness. All ten generated sequences
match within each configuration; agreement between quantization recipes is
not required or claimed.

- **TTFT:** from pretokenized input and empty, synchronized GPU KV until the
  first greedy token is available on CPU. Includes batch construction,
  synchronized prefill, last-position logits transfer, finite check and CPU
  argmax. Excludes loading, tokenization, initial KV reset, queuing and
  networking. P50/P95 use NumPy's default linear percentile interpolation on
  ten request latencies, not a service-level latency distribution.
- **Prefill TPS:** total input tokens divided by summed synchronized prompt
  evaluation times. Includes C++ batch construction; excludes finite check
  and first-token selection. Only the last prompt position produces output
  logits. Inputs above 2048 tokens are submitted in consecutive chunks of at
  most 2048, each with synchronized execution; microbatch remains 512.
- **Decode TPS:** 1270 tokens divided by summed decode wall time for ten
  requests, excluding the first token produced by prefill. Every request
  retains its full input context. Includes C++ orchestration, batch updates,
  synchronized GPU execution, logits transfer, finite check and CPU argmax.
- **Process Peak:** NVML GPU process-resident bytes during measured requests
  only, after three warmups, including retained weights, KV, workspace and
  allocations. A ready/go handshake excludes loading and warmup from the
  sampling window. Requested interval is 5 ms; observed maximum gaps range
  from 11.69 to 23.34 ms across configurations. This is not host RAM or an
  exact allocator high-water mark, and brief peaks can be missed.
- KV is cleared and synchronized before every request, outside TTFT. Requested
  context capacity is input plus 128; actual capacities are rounded to
  256/768/2304/4352 for the four input lengths respectively.
- GPU 6 must have no compute processes before each load. Sampling observes
  exactly one compute PID per run; NVML uses the host PID rather than the
  Docker subprocess PID. Runs with additional observed PIDs are rejected.
  Other GPUs and shared-host CPU/power/thermal conditions are not controlled.

## Results

| Input Tokens | Model | TTFT P50 (ms) | TTFT P95 (ms) | Prefill TPS | Decode TPS | Process Peak (GiB) |
|---:|---|---:|---:|---:|---:|---:|
| 128 | Q4_K_M | 100.29 | 102.68 | 1274.65 | 159.42 | 17.80 |
| 128 | Q2_K | 160.28 | 162.37 | 799.87 | 144.29 | 11.07 |
| 512 | Q4_K_M | 129.64 | 130.15 | 3967.50 | 158.55 | 18.01 |
| 512 | Q2_K | 189.96 | 192.34 | 2698.23 | 143.58 | 11.28 |
| 2048 | Q4_K_M | 435.29 | 438.18 | 4708.89 | 155.39 | 18.16 |
| 2048 | Q2_K | 623.53 | 627.82 | 3282.36 | 140.45 | 11.43 |
| 4096 | Q4_K_M | 839.67 | 843.17 | 4879.22 | 150.98 | 18.35 |
| 4096 | Q2_K | 1180.07 | 1183.92 | 3470.47 | 136.67 | 11.62 |

Changes are calculated from unrounded values, with Q4_K_M as the reference.

| Input Tokens | Q2_K TTFT P50 Increase | Q2_K Prefill TPS Change | Q2_K Decode TPS Change | Process Peak Saving (GiB) |
|---:|---:|---:|---:|---:|
| 128 | +59.81% | -37.25% | -9.49% | 6.7305 |
| 512 | +46.53% | -31.99% | -9.44% | 6.7305 |
| 2048 | +43.24% | -30.29% | -9.61% | 6.7305 |
| 4096 | +40.54% | -28.87% | -9.48% | 6.7305 |

Q2_K has lower sampled process residency at every length, but higher TTFT
and lower prefill/decode throughput in this run. No isolated-kernel explanation,
confidence interval, accuracy preservation or general hardware speedup is claimed.

## Model Size and GGUF Size

All GiB values use 2^30 bytes. **Model tensor size** is the serialized tensor
payload reported by `llama_model_size`, including quantization scales and
other stored tensor data, but excluding GGUF metadata/alignment. **GGUF size**
is the actual file size. Neither is runtime GPU residency.

| Model | Stored Parameters | Model Tensor Size (bytes) | Model Tensor Size (GiB) | GGUF Size (bytes) | GGUF Size (GiB) |
|---|---:|---:|---:|---:|---:|
| Q2_K | 30,532,122,624 | 11,252,639,744 | 10.4798 | 11,258,610,240 | 10.4854 |
| Q4_K_M | 30,532,122,624 | 18,550,716,416 | 17.2767 | 18,556,686,912 | 17.2823 |

Each GGUF has 5,970,496 bytes of non-tensor file overhead. Q2_K is 39.33%
smaller than Q4_K_M by GGUF file size. For context, storing this parameter
count uniformly at two bytes per parameter would take 56.8705 GiB; this is
an arithmetic FP16/BF16-equivalent estimate, not a measured checkpoint file.

The ORT models used in the original report have the following measured local
storage sizes. ONNX totals count the graph plus each uniquely referenced
external-data file once; tokenizer/configuration files are excluded. These
are different quantization recipes, not matched encodings or quality levels.

| Runtime / Recipe | Model File(s) | Graph (bytes) | External Weights (bytes) | Total Model Storage (GiB) |
|---|---|---:|---:|---:|
| ORT INT4 block64 | model.onnx + model.onnx.data | 900,056 | 16,405,168,128 | 15.2793 |
| ORT mixed INT2/INT4 block64 | model.onnx + model.onnx.data | 902,868 | 13,989,249,024 | 13.0293 |
| llama.cpp Q4_K_M | single GGUF | N/A | N/A | 17.2823 |
| llama.cpp Q2_K | single GGUF | N/A | N/A | 10.4854 |

## Actual Quantization Recipes

The following mapping was read from the downloaded GGUF headers, not inferred
from the filename. Layer counts refer to zero-based layers 0 through 47.

| Tensor Family | Q2_K File | Q4_K_M File |
|---|---|---|
| Expert gate | Q2_K, all 48 layers | Q4_K, all 48 layers |
| Expert up | Q2_K, all 48 layers | Q4_K, all 48 layers |
| Expert down | Q3_K, all 48 layers | Q6_K in 24 layers; Q4_K in the other 24 |
| Attention Q / K | Q2_K, all 48 layers | Q4_K, all 48 layers |
| Attention V | Q4_K, all 48 layers | Q6_K in 24 layers; Q4_K in the other 24 |
| Attention output | Q3_K, all 48 layers | Q4_K, all 48 layers |
| Token embeddings | Q2_K | Q4_K |
| LM head | Q6_K | Q6_K |
| MoE router and normalization weights | F32 | F32 |

The Q4_K_M Q6_K down/V layer indices are
`0,1,2,3,4,5,8,11,14,17,20,23,26,29,32,35,38,41,42,43,44,45,46,47`.
Both files contain 579 tensors. Type counts are:

- Q2_K file: 241 F32, 193 Q2_K, 96 Q3_K, 48 Q4_K, 1 Q6_K.
- Q4_K_M file: 241 F32, 289 Q4_K, 49 Q6_K.

Neither file is uniformly two-bit or four-bit. Q2_K uses two-bit expert
gate/up in all 48 layers, but three-bit expert down in all 48 layers. The
ORT mixed recipe instead uses uniform symmetric block64 INT2 gate/up in
24 layers, with remaining expert projections INT4; the ORT INT4 model has
all expert projections INT4. Both ORT recipes use INT4 attention/embeddings,
INT8 head and FP16 graph/activations, as described in [README.md](README.md)
and [PERFORMANCE.md](PERFORMANCE.md).

GGUF K-quants include their own blockwise scales and encoding. Their nominal
bit labels are not total bits per stored parameter, and they are not the
ORT packed INT2/INT4 ABI. This experiment does not isolate gate/up bit width
from down-projection, attention or embedding quantization differences.

## Comparison Boundaries

The column layout and prompt are aligned with the ORT report, but these are
not controlled same-quantization cross-runtime trials. The ORT report used
GPU 1 on October 4, Python orchestration, no CUDA capture, full-sequence
prefill logits and different weight recipes. This report uses GPU 6 on
October 9, C++ orchestration, default CUDA Graph behavior and only final
prompt-position logits. Prefill batching and KV capacity also differ.
Identical prompt IDs do not establish checkpoint/tokenizer provenance or
quality equivalence. No llama.cpp-versus-ORT speedup or accuracy claim follows.

P95 is based on only ten requests per configuration. Greedy continuations
can differ between quantization recipes and change expert routing. No task
accuracy, concurrency, service TTFT or other GPU family was evaluated.

Earlier local `llama-bench` measurements used depth-zero generation and a
process peak spanning loading/warmup/all test cases. They are excluded here.
The combined `-pg` metric is not context-dependent pure decode TPS and is
not used to fill this table.

## Local Reproduction Artifacts

The validation machine retains `qwen3_llama_request_benchmark.cpp`, its linked
executable, and `qwen3_llama_request_compare.py` under
`/datadisks/disk1/jiafa/accuracy/`. Final raw per-configuration JSON/JSONL/log
files and `summary.json` are in `qwen3-llama-request-quiet-perf-20261009/`.
The smoke runs and earlier runs with measurement-period DEBUG logging are
excluded. Harnesses, binaries, GGUFs and raw results remain local and are not
distributed in this documentation PR.

SHA256 fingerprints:

| Artifact | SHA256 |
|---|---|
| Qwen3-30B-A3B-Q2_K.gguf | `db3ce897ccc9e7d9dbf17fe083cae7880a2092aa473b45eba8b77715aa9ca170` |
| Qwen3-30B-A3B-Q4_K_M.gguf | `9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48` |
| qwen3_llama_request_benchmark.cpp | `50ffa6fe6ad3ab5862dd78e5c8b0ecdabc20f110da9aad6d2b42391d4a61d6d6` |
| qwen3_llama_request_compare.py | `ac672774336eb5c4197314bb74110dfe8c18f73c8eee12699d5cfe133b67ede5` |
| qwen3_llama_request_benchmark executable | `13103e48d5e3e0c889054a54e51846ee3417b0371813e615d35e5f3412b7e483` |

On that machine, using the retained harnesses and matching build:

```bash
docker exec jiafa-dev \
  /datadisks/disk1/jiafa/accuracy/onnxruntime/.venv/bin/python \
  /datadisks/disk1/jiafa/accuracy/qwen3_llama_request_compare.py \
  --output-dir /datadisks/disk1/jiafa/accuracy/qwen3-llama-request-rerun
```