# Experimental Nemotron 3.5 Lightning BF16 target + MTP export (CUDA)

This Olive recipe uses
[Mobius PR #761](https://github.com/onnxruntime/mobius/pull/761) to construct
both target and draft ONNX graphs for
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).
It targets **CUDAExecutionProvider only**.

**Full-model numerical parity remains blocked.** PR #761 reports 141/240
decoder comparisons failing in the supplied H100 evidence. Short matching
token sequences and conditional MTP agreement are not full-model correctness.
This contribution adds an experimental Olive adapter, not a validated export,
inference package, speedup, or implemented speculative-generation loop.

## Pins and scope

| Input | Immutable revision |
|---|---|
| NVIDIA BF16 checkpoint | `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` |
| Mobius exporter | `e3d01a7712f38deb962e7107752da7fb668a7054` |
| Olive | `18bf0b7f4943cd6ca7a8546a4c427e34d7343543` |

Mobius PR #761 is not merged at this exporter pin. The checkpoint declares
52 hybrid Mamba/attention/MoE layers and one NextN prediction step. The
explicit `nemotron-h-mtp` task preserves target and MTP weights; this recipe
does not silently remove `mtp.*` tensors or select target-decoder-only mode.
Native NVFP4 is unsupported and is not a variant of this recipe.

The stock Olive `MobiusBuilder` at the pinned version cannot select this task
and writes generic GenAI metadata. A small local pass therefore calls the
public Mobius builder with the explicit task and returns an Olive
`CompositeModel` containing `decoder` and `mtp`, preserving their directories
and package-level reports. It deliberately does **not** generate a
`genai_config.json` or pretend the recurrent-cache/runtime ABI is validated.

## Development export

Run from the `nvidia-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/` directory.
Prepare an isolated development environment and install:

```bash
python -m pip install -r cuda/requirements.txt
```

The two source revisions are pinned; transitive dependencies and the actual
CUDA/ORT build are **not** an approved environment lock. Establish compatible
BF16 kernels, host-memory/disk budgets and a matching NVIDIA driver/CUDA
runtime before any full export. The checkpoint index declares approximately
65.8 GB of tensor data before loader, graph, cache and serialization overhead.
The 3B active-parameter label does not mean 3B resident weights; a 24 GB RTX
4090 is not claimed to hold the unquantized model.

The checked-in configuration refuses weighted export by default. After
explicitly accepting the unresolved numerical gate:

```bash
python cuda/run.py --allow-unvalidated-export
```

This command downloads the pinned weights and can require substantial
resources. It was **not run** as part of this contribution. The wrapper
registers the local pass and enables the development-only opt-in in memory;
it does not modify the source configuration and fails if Olive produces no
output model. Do not substitute a bare
`olive run --config cuda/config.json`: the custom pass requires its package
registration and the default gate is intentionally closed.

Olive places the final composite under `cuda/models/` and its work cache
under `cuda/cache/`. Preserve the exported `decoder/model.onnx`,
`mtp/model.onnx`, external tensor data and `weight-loading-report.json`
together. A cached successful export may be reused by Olive; for a fresh
export, use fresh output/cache paths rather than overwriting previous results.

## Runtime and acceptance boundaries

The target produces post-final-normalization `mtp_seed`. Pair target hidden
state `h_i` with token `t_(i+1)` to predict `t_(i+2)` in the draft; draft KV is
independent of target KV. A runtime must implement target verification,
accepted-prefix commit and rejected-state restore/replay for draft KV and
target attention KV **and** Mamba convolution/SSM state. This recipe implements
none of that coordination and provides no generic GenAI inference command.

Before declaring the package correct, validate every source tensor, full
prefill/cached logits, convolution/SSM/KV state updates and reset, deterministic
multi-token generation, external data and actual CUDA node placement using the
unchanged upstream acceptance thresholds. Conditional MTP success alone is
insufficient. See the
[pinned export contract](https://github.com/onnxruntime/mobius/blob/e3d01a7712f38deb962e7107752da7fb668a7054/docs/nemotron-lightning-export.md).

## Offline tests

These tests use the real pinned Olive schemas, pass API and workflow/cache
publication with a fake Mobius package and mocked provider discovery. They
do not download weights, build a real model or execute CUDA:

```bash
python -m pip install -r cuda/requirements-test.txt
OLIVE_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest cuda/lightning_builder_test.py -q
```

No `olive_ci.json` is added: automatic full-size GPU export is not appropriate
while the numerical gate remains blocked.
