# Reproduction and evidence boundaries

## CPU-only checks

From this model directory, with Python 3.12:

```bash
python3 scripts/evidence.py verify --check-docs
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/evidence.py tables --section accepted
python3 scripts/evidence.py tables --section allocator-7k
python3 scripts/evidence.py tables --section allocator-28k
python3 scripts/evidence.py tables --section qmoe
python3 scripts/evidence.py tables --section weights
python3 scripts/evidence.py tables --section logits
python3 scripts/evidence.py tables --section sequential
python3 scripts/evidence.py summary > summary.json
```

These use only the standard library. They reconstruct statistics and
calculations, not fresh GPU measurements. The accepted CSV contains all
135 records, logical worker identities and unrounded saved timing metrics.
The extraction was checked against raw CSV/JSONL and every token fixture
before removing machine identifiers.

`evidence/SHA256SUMS` covers every compact input. This detects accidental
changes; it is not a cryptographic attestation of the original hardware run.
Logical worker IDs preserve grouping, while actual process-ID independence
was validated from the retained raw source before publication projection.

[Model-local Git attributes](.gitattributes) force LF for the evidence inputs,
the checksum manifest and the hash-pinned runner, including with
`core.autocrlf=true`. Do not regenerate historical checksums to accommodate
converted CRLF files. Restore their original LF bytes instead. The
checkout/filter regression builds a disposable Git repository, so it also works
in archive copies without a surrounding `.git`; it is skipped only if Git is
absent. CPU arithmetic verification itself needs only the standard library.

The verifier cross-checks accepted declarations against CSV-derived counts,
variants, contexts/capacities and repetition/request structure. It rejects
missing or contradictory success/profiling/GPU-0 declarations. Per-request
GPU/profiling facts are not guessed or added to the historical CSV: consistency
checks do not independently prove physical GPU identity or process isolation.

The two logits pair CSVs retain exact unequal FP16 bit-pair counts, including
any signed-zero bit distinction. Equal pairs are tallied separately.
For finite value bits `b`, the ordered key is:

```text
b has sign bit: 0x8000 - (b & 0x7fff)
otherwise:      0x8000 + (b & 0x7fff)
steps:          abs(key(original) - key(candidate))
mean abs:       sum(count * abs(value(original) - value(candidate))) / 151936
```

The projection is sufficient for the reported difference metrics, but cannot
recover vocabulary order or the full logits vectors. This is not an
independent rerun of the MatMul or proof of model quality.

## Pinned identities

[Provenance](evidence/provenance.json) records:

- Hugging Face model/revision, quantization settings and executed pass lineage.
- Runtime package versions, exact ORT/GenAI source commits and artifact hashes.
- Original and candidate graph identity; external-data identity.
- GGUF repository/revision, size, download etag and later SHA-256.
- Source evidence fingerprints and projection descriptions.

The large weight files were not hashed by the original benchmark. Later
integrity checks hashed them; those later fingerprints do not retroactively
prove build-time weight bytes. The candidate shared the original external
data inode and only changed its copied small graph.

Unknowns are explicit:

| Missing identity/check | Consequence |
|---|---|
| Historical Olive/Mobius versions | Not recovered. One fresh export with the pinned sources reproduced the archived external-data bytes (single run; see [Findings](FINDINGS.md#fresh-export-and-mmlu-sanity-check)); this does not identify the historical versions |
| Separate native llama.cpp commit | Binding version/file fingerprints available; native revision not independently pinned |
| Every build's CUDA toolkit | `libcudart.so.13` is an observed runtime-library name, not a compiler-toolkit identity |
| Public distribution URL for the exact original ONNX package | GPU reproduction needs supplied matching artifacts |
| Repeat runs of the fresh export | One fresh export (2026-10-08) was hashed and smoke-tested; timings and memory are single runs |

The current build source pins follow the existing base-model recipe. They
are a proposed setup, not evidence of which export environment ran in
September. Report new export hashes/version changes instead of substituting
the new output silently into these historical results.

## Optional GPU setup and baseline export

GPU commands below are **not run** by the CPU verification. They were executed
once on 2026-10-08, after this PR was opened (see
[Findings](FINDINGS.md#fresh-export-and-mmlu-sanity-check)). Use a Linux CUDA
machine, keep GPU 0 isolated, and preserve original models. On that date the
approved feed served the pinned `onnxruntime-gpu` and `onnxruntime-genai-cuda`
wheels; exact matching runtime wheels may still need source builds if it does not.

Run from this model directory:

```bash
export PIP_INDEX_URL="https://packagefeedproxy.microsoft.io/pypi/simple"
unset PIP_EXTRA_INDEX_URL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r cuda/requirements.txt

export BENCHMARK_GPU_INDEX=0 CUDA_VISIBLE_DEVICES=0
olive run --config cuda/kquant_fp16/config.json
```

Output is `cuda/kquant_fp16/models/`. No hidden-state-row selection is added
by this baseline recipe. In the 2026-10-08 run the fresh external data was
byte-identical to the archived file, and `model.onnx` differed only in its
`graph.name` string (an absolute Olive-cache path); see
[Findings](FINDINGS.md#fresh-export-and-mmlu-sanity-check). `requests` is listed
in `cuda/requirements.txt` because the pinned Olive commit imports it without
declaring it. These are single-run facts, not a determinism claim.

For source-built measured runtimes, use the pinned repositories rather than
an arbitrary current release:

```bash
git clone https://github.com/microsoft/onnxruntime.git ort-source
git -C ort-source checkout f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7
git clone https://github.com/microsoft/onnxruntime-genai.git genai-source
git -C genai-source checkout 83de55a1f3886cddbb32a5517e26adb3f9d65159
```

Follow the corresponding pinned build instructions for the platform/toolchain
and install those wheels into the environment. A universal one-command native
build is not supplied: compiler/toolkit provenance was not fully recovered.
If a build needs NuGet, the approved feed is
`https://packagefeedproxy.microsoft.io/nuget/v3/index.json`.

## Optional accepted-workload benchmark reproduction

The included [runner](scripts/run_ort_llama_benchmark.py) is a publication copy
of the corrected measurement harness, with reproducibility fixes for provider
metadata, physical GPU identity and platform boundaries. It is **not relabeled
as the historical runner**. Historical SHA/AST identities remain unchanged
in [protected provenance](evidence/provenance.json); revised file identity and
the exact named algorithms compared are recorded separately in
[runner lineage](scripts/runner_lineage.json), outside protected evidence.

Actual AST comparisons cover prompt/capacity construction, token/EOS counting,
timing arithmetic and boundaries, runtime request loops, sample acquisition/
window definitions, the independent sampler and fresh-state checks. Platform/
UUID startup helpers, trial-worker preflight and provenance collection changed;
the whole-runner measurement AST is therefore not claimed identical.
The superseded pasted wrappers are not used.

The declared `nvidia-ml-py` distribution provides the `pynvml` import namespace.
Future-run provenance requires that provider's metadata; a separately installed
legacy `pynvml` distribution is explicitly optional and records null when absent.
Other required package-metadata failures remain errors. Historically recorded
versions, including legacy `pynvml`, are not rewritten.

Supply the original runtime and artifact identities; do not rebuild or
substitute artifacts silently. Install the separate benchmark dependencies
only when intentionally running this GPU workflow:

`llama-cpp-python` must be CUDA-enabled; its version alone does not establish
the native build's backend or source commit. For an intentional source install,
set `CMAKE_ARGS="-DGGML_CUDA=on"` and `FORCE_CMAKE=1` when installing that
requirement. A CPU-only wheel is not a reproduction of the recorded run.

```bash
python -m pip install -r cuda/requirements-benchmark.txt
: "${ORT_MODEL:?Set the original ORT GenAI artifact directory}"
: "${GGUF_MODEL:?Set the matching Q4_K_M GGUF file}"
export BENCHMARK_GPU_INDEX=0 CUDA_VISIBLE_DEVICES=0
nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv
```

The runner refuses unrelated GPU processes, holds the GPU0 run lock, verifies
the original config and creates capacity/allocator overlays without writing
through config hardlinks. It stops on an error and never overwrites existing
benchmark outputs.

GPU execution is intentionally Linux-only; `fcntl` is imported only in the
guarded Linux locking path, after platform rejection and before Unix-only
operations. CPU helper imports and `--help` do not need Unix modules or GPU
packages. Linux simulations of missing `fcntl` are not native Windows validation.

The physical-device policy is **NVML index 0**, not assumed CUDA ordinal 0.
Before either inference runtime imports, the resolver sets
`CUDA_VISIBLE_DEVICES` to the full UUID of that device and exports
`QWEN_BENCHMARK_GPU_UUID` to workers/samplers. Unset or numeric `0` visibility
is canonicalized; a matching full UUID is valid; conflicting, empty or
multiple-device visibility is rejected. `CUDA_DEVICE_ORDER` alone is not proof.

The trial worker checks logical CUDA device 0's UUID with driver identity APIs
(`cuInit`, device count/device/UUID queries), without context-creation or GPU
allocation calls. NVML contexts revalidate the selected physical UUID. The
query occurs in worker startup before runtime import, model-load timing and
the independent sampler baseline; it adds preflight CPU/driver work, not
changes to existing load/inference timing boundaries or sample definitions.
NVIDIA documents that
[`cuInit` may preload JIT libraries](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html).
That initialization cost can be relocated into preflight compared with the
historical runner. Model-load timing comparisons require separate accounting
of this revised startup; no new performance equivalence is claimed.
Empty NVML process lists remain legitimate before initialization and after
cleanup. New records separately identify execution and monitoring UUIDs.

This pass exercises the identity APIs only through mocks. Actual native
CUDA/NVML startup behaviour, GPU-memory effects and native Windows Python 3.12
remain unexecuted boundaries; no claim is made that historical GPU selection
was incorrect.

For the exact workload matrix:

```bash
OUTPUT="diagnostic-output/matrix_$(date -u +%Y%m%d_%H%M%S)"
python scripts/run_ort_llama_benchmark.py \
  --ort-model "$ORT_MODEL" --gguf-model "$GGUF_MODEL" --output-dir "$OUTPUT" \
  --contexts 2048,4096,8192,16384,32768 --output-tokens 64 --repetitions 3 \
  --runtime both --ort-device-allocator --sampling-interval-ms 10 --cold-warm
```

`--contexts` accepts **nominal labels**. Labels 8192/32768 produce the
recorded 7189/28747 token counts only with the recorded prompt construction
and tokenizer. `--ort-device-allocator` adds the Reserve variant; it does
not remove the baseline variant. Summarize them separately.

The exact execution-time command used an activated VM environment and absolute
locations. The command above is its portable equivalent, not a fabricated
byte-for-byte execution-time shell string.

For a focused original-versus-candidate memory diagnostic, supply a separately
prepared complete candidate artifact and run each into a fresh directory:

```bash
: "${CANDIDATE_MODEL:?Set the isolated candidate ORT artifact directory}"
for variant in original candidate; do
  model="$ORT_MODEL"
  if [ "$variant" = candidate ]; then model="$CANDIDATE_MODEL"; fi
  OUTPUT="diagnostic-output/${variant}_$(date -u +%Y%m%d_%H%M%S)"
  python scripts/run_ort_llama_benchmark.py \
    --ort-model "$model" --gguf-model "$GGUF_MODEL" --output-dir "$OUTPUT" \
    --contexts 8192,32768 --output-tokens 64 --repetitions 1 --runtime ort \
    --ort-device-allocator --sampling-interval-ms 10 \
    --ort-profile-dir "$OUTPUT/profiles"
done
```

These profiled timings are **diagnostic only**; do not pool them with the
unprofiled accepted matrix. The runner does not capture paired last-token
logits or both allocator instances. It is not a replacement for those
instrumented measurements and cannot establish candidate acceptance.

## Instrumented-diagnostic reproduction limits

Direct-counter captures used a standalone C++ adapter compiled against the
exact GenAI/ORT headers, reading actual model-session and global allocator
instances through `AllocatorGetStats`. The captured current/requested/held/
high-water counters and all token comparisons are committed in compact form.

The internal-ABI adapter, compiled binary, raw profiles and historical
VM-specific scripts are intentionally not bundled as a generally supported
public API. Thus:

- Every published **calculation** can be reproduced on CPU from this PR.
- Fresh direct-counter capture or paired-logits capture requires an
  appropriately instrumented executable/environment; this PR does not
  provide a turnkey portable reproduction of those native captures.
- No arena-disabled, shrink, allocation-stack, numerical-repeatability or
  varying-shape control is claimed to have been executed here.

Ask the relevant runtime owner about supported instrumentation before
extending that path. Preserve normal-allocator baselines.

## Safety and unsupported environments

Before/after SHA-256 manifests protect original graphs/configs/weights,
accepted results and previous diagnostics. Original files are never inputs
to an in-place rewrite. Share artifact fingerprints, not weight contents.

The native diagnostics were captured on Linux x86_64/A100 CUDA with the
pinned packages. Windows, other EPs, different GPUs, different runtime/source
versions, ragged batching, speculative decoding and other quantizers have
not been validated by this investigation.
