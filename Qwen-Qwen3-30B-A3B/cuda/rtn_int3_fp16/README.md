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
reshape/transpose nodes, which an ORT optimizer can fold. The configured output
directory is `models/qwen3-30b-a3b-rtn-int3-fp16`.

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

The report retains measured results, test conditions, implementation differences,
and qualification limits. Experiment harnesses, tests, raw captures, and build
logs are not part of this recipe. The reported measurements reused an existing
audited export; they do not establish reproducibility from this recipe alone.
The full-model comparison is not an accuracy evaluation, quantization-quality
claim, or proof of numerical equivalence between the implementations.