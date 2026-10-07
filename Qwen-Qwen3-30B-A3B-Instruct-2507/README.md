# Qwen3-30B-A3B-Instruct-2507: KQuant/Mobius recipe and memory findings

This contribution records a successfully exported
[`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
baseline and a reproducible GPU-memory investigation. **It does not implement
or claim an accepted memory optimization.**

The original model, the last-row logits candidate, and llama.cpp answer
different questions. The corrected runtime comparison is kept separate from
instrumented diagnostics and numerical-equivalence experiments.

## Recipe and scope

[The recipe](cuda/kquant_fp16/config.json) runs Olive KQuant with 4-bit
symmetric groups of 128, `moe=true`, `embeds=false`, and `lm_head=false`,
followed by MobiusBuilder with `precision=fp16`. The model revision is
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.

This is a separate model ID from the original `Qwen/Qwen3-30B-A3B` discussed in
[PR #637](https://github.com/microsoft/olive-recipes/pull/637). It does **not**
establish that #637 is incompatible with Instruct-2507, and does not copy the
base-model quality results.

The recorded Olive pass lineage confirms that MoE quantization actually ran;
QMoE operators alone are not used as proof. The authored recipe was untracked.
Historical Olive/Mobius versions were not recovered: the proposed source pins
in [requirements](cuda/requirements.txt) follow the current base-model recipe
and are **not asserted to be the historical export environment**.

## Results at a glance

The accepted benchmark has **135 successful requests, 45 independent workers,
three repetitions per group**, five context tiers and three variants. Cold,
warm_1 and warm_2 are separate populations; no requests are hidden as warmups.

The first three rows below are cold-request medians of three workers.
The last row is a **separate diagnostic with one run per context**.
All values are sampled NVML **device-used** peaks in MiB, not GiB; load-inclusive
per-request peaks are used consistently. This is not a quality-equivalent
runtime ranking.

<!-- BEGIN headline -->
| Sampled device-used peak | 7,189 prompt tokens | 28,747 prompt tokens |
|---|---:|---:|
| ORT baseline, accepted median | 39,388.88 MiB | 60,882.88 MiB |
| ORT Reserve, accepted median | 25,222.88 MiB | 47,238.88 MiB |
| llama.cpp, accepted median | 19,618.88 MiB | 24,178.88 MiB |
| Last-row candidate, diagnostic n=1 | 21,126.88 MiB | 30,854.88 MiB |
<!-- END headline -->

Full timing/memory medians and observed ranges for all 45 groups are in
[the findings](FINDINGS.md#accepted-benchmark). They regenerate from
[135 compact request records](evidence/accepted_requests.csv).

### Main findings

- **QMoE workspace is reused across layers.** Layer 0 establishes the
  high-water requirement; layers 1-47 add no arena-reservation growth.
  A source-based buffer subtotal of 1,212.14 / 4,847.02 MiB matches the
  observed increase of 1,212.15 / 4,847.05 MiB, apart from small alignment
  and metadata terms. This does not establish that every buffer is optimal.
- **Allocator ownership matters.** Model intermediates use the model-session
  allocator. KV cache, raw logits and other device buffers use a separate
  GenAI global allocator, which survives request and model destruction.
- **Full-sequence logits are a measured opportunity, not an accepted fix.**
  Selecting the last hidden row before the LM head saves 4,096 / 16,384 MiB
  at 7,189 / 28,747 tokens. All 64 output IDs matched, but logits were not
  exact: maximum absolute difference 0.015625, ordered-FP16 distances
  188 / 580 near zero. The cause and acceptance criterion remain unresolved.
- **The single-request budget is bounded, not completely identified.**
  Direct counters account for 99.54% / 99.76% of peak process memory.
  The remaining 115.10 MiB is bounded by lifetimes, not uniquely assigned
  to CUDA/library/driver owners.
- **Sequential retention is a separate open issue.** Fifteen identical
  requests in one model/session showed +86 MiB after cleanup, no plateau,
  constant allocator capacities, and zero GenAI live/requested bytes after
  every request. All 960 output IDs matched. The growth's owner is unknown;
  this is not a demonstrated tensor leak or proof of indefinite growth.

## Verify without a GPU or this VM

From this model directory, Python 3.12 with the standard library is sufficient:

```bash
python3 scripts/evidence.py verify --check-docs
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/evidence.py tables
```

The verifier reconstructs benchmark statistics, same-checkpoint byte budgets,
QMoE deltas, token comparisons and FP16 metrics from the committed evidence.
It does not import CUDA, ORT GenAI, llama.cpp, NumPy or model data.

Accepted provenance is cross-checked against CSV-derived counts, variants,
contexts/capacities, repetitions and cold/warm structure. Successful exit,
unprofiled and physical-GPU-0 declarations are also required. This is
declaration consistency, not an independent re-observation of historical GPU
identity, profiling or process isolation. Model-local Git attributes keep
the evidence inputs, checksum manifest and hash-pinned runner in LF form.

| Read next | Purpose |
|---|---|
| [Findings](FINDINGS.md) | Full tables, measurement definitions, causal limits and open decisions |
| [Reproduction](REPRODUCING.md) | Dependencies, source pins, optional GPU commands and unsupported paths |
| [Provenance](evidence/provenance.json) | Build lineage, artifact fingerprints, versions and projection limits |
| [CPU evidence tool](scripts/evidence.py) | Exact calculations and fail-fast validation |
| [Publication runner lineage](scripts/runner_lineage.json) | Historical identities, revised copy identity and named unchanged algorithms |

## Limitations and next decisions

ONNX KQuant and GGUF Q4_K_M are different quantized artifacts; the GGUF uses
mixed Q4_K/Q6_K storage. Same inputs, capacities and output lengths do not
establish numerical or output-quality equivalence. No MMLU/quality result is
claimed for this artifact.

The experimental candidate's provisional gap to the existing llama.cpp
baseline is **1.47 GiB at 7K and 6.52 GiB at 28K**. The additional 5.05 GiB
context-dependent difference is not closed by the internal accounting
residual. Allocation types/lifetimes still need comparison between runtimes.
These are diagnostic-to-benchmark comparisons, not fresh paired candidate
benchmark medians.

Explain the logits differences before proposing exactness or a tolerance.
Coordinate the Mobius integration with existing last-token-logits work,
including [GenAI PR #2393](https://github.com/microsoft/onnxruntime-genai/pull/2393),
whose paged-Engine path does not automatically rewrite this existing
non-paged Mobius artifact. Repeated-request growth requires separate owner
attribution. No runtime fixes, arena-disable recommendation, expert-offloading
implementation or optimization-acceptance decision is included here.
