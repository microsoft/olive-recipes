# All-expert INT3 RTN, FP16, block64

This experimental recipe exports Qwen3-30B-A3B with **all routed expert
gate, up, and down weights in all 48 layers quantized to INT3**. Attention and
token embeddings remain INT4, the language-model head is INT8, and the graph,
activations, scales, and KV cache use FP16. Routers and normalization parameters
remain floating point. Symmetric RTN uses block/group size 64 and needs no
calibration dataset.

The checkpoint is pinned to
`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
This is not a uniformly three-bit model: non-expert weights use the precisions
above. It is a separate recipe, not the selective INT2/INT4 layer schedule from
[olive-recipes #648](https://github.com/microsoft/olive-recipes/pull/648).

## Export

From the `Qwen-Qwen3-30B-A3B` directory:

```bash
python -m pip install -r cuda/rtn_int3_fp16/requirements.txt
olive run --config cuda/rtn_int3_fp16/config.json
python cuda/rtn_int3_fp16/audit_model.py /path/to/exported/package \
  --output /path/to/model-audit.json
```

Use an isolated environment. The requirements pin experimental INT3 producer
commits from [Olive #2749](https://github.com/microsoft/Olive/pull/2749) and
[Mobius #783](https://github.com/onnxruntime/mobius/pull/783). Installing these
requirements does **not** install either experimental ORT CUDA implementation.
Build and select the desired runtime separately. No released-runtime or ORT
GenAI execution qualification is implied.

Canonical weights use row-local LSB-first three-bit packing with offset-4
symmetric codes. The model has `weights_prepacked=0`; it is not an offline SM80
INT2/INT4-prepacked model. FC1 gate/up interleaving is represented by static
reshape/transpose nodes, which an ORT optimizer can fold. The audit traces these
connections to source initializers without loading the external weights into RAM.
It verifies all 48 layers, effective bit widths, shapes, FP16 scales, the
attention/head/embedding recipe, external-data ranges, and streaming SHA256 hashes.
This structural check does not independently reconstruct every packed weight.

## Runtime Comparison

[PERFORMANCE.md](PERFORMANCE.md) records the same-artifact comparison of:

- [ORT #33555](https://github.com/microsoft/onnxruntime/pull/33555),
  `d75ac16f377c0c783d289046f5cf2efe6bb5f9bc`;
- [ORT #33569](https://github.com/microsoft/onnxruntime/pull/33569),
  `4ba6c38384cf5b12d4aa4570cd15af2461c4d515`.

Both snapshots share main base `36f07f760dccb74a96d0a3c851163d6448bdfd21`.
Build each independently in Release mode with CUDA 13, SM80, Python bindings,
and the shared library enabled. Use the same compiler, build flags, dependency
source revisions, and Python environment. Do not replace a provider in an
unrelated cached ORT installation: its internal ABI may differ.

The comparison driver requires two complete Python package roots, each
containing `onnxruntime/capi/onnxruntime_pybind11_state.so` and its matching
provider libraries. It hashes the actually mapped binaries and checks the
model, prompt, forced tokens, and GPU identities across runs.

```bash
python cuda/rtn_int3_fp16/run_comparison.py \
  --model-dir /path/to/exported/package \
  --runtime33555 /path/to/pr33555/python-package-root \
  --runtime33569 /path/to/pr33569/python-package-root \
  --physical-gpu 0 --output-dir /path/to/new-results
```

The driver runs sequentially on one initially idle GPU. It first captures a
128-token greedy continuation from #33569, then checks #33555 on the same token
stream. Timing uses this fixed stream for both implementations, not independent
greedy contexts. The default matrix is batch one, input lengths
128/512/2048/4096, 128 output predictions, two alternating-order rounds,
three warmups and five measured requests per process: ten measured requests per
runtime/length. EOS is ignored. Different input lengths share the fixed decode
stream; longer-prompt continuations are synthetic teacher-forced workloads.

CUDA graphs, ORT profiling, and QMoE debug switches are disabled. CPU shape
operations are allowed, but CUDA provider loading is required. The scratch
setting is 64 MiB per QMoE operation, not a whole-process memory limit. Session
initialization and NVML-sampled residency are reported separately. Last-position
logits are copied to CPU for finite checks, argmax, and hashing. This is a
full-model request benchmark, **not** isolated kernel timing or a service load test.

For a single natural-greedy request, use `benchmark.py` without `--token-stream`;
see `--help` for required artifact/runtime commit labels and output options.
`compare_logits.py` reports numerical differences; nonzero exit status on
allclose failure must not be interpreted as proof of an individual kernel bug.

## Validation And Limits

```bash
python -m pytest cuda/rtn_int3_fp16/ -q
```

Tests cover the full layer schedule, graph audits, unsafe external paths,
static interleave chains, token counts, timing aggregation, paired-run identity,
and logits comparison. Tests do not substitute for CUDA correctness tests in ORT.
The full-model comparison is not an accuracy evaluation, quantization-quality
claim, or proof of numerical equivalence between the implementations.