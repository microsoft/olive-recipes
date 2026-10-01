# KEV-4B CUDA Mobius export

This recipe exports `jaredpalmer/kev-4b` as a multi-component
non-generative package for Foundry Local.

## Setup

From the `olive-recipes` repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r kev-4b/cuda/requirements.txt
```

Download the pinned adapter and head:

```bash
hf download jaredpalmer/kev-4b \
  --revision 139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --include adapter_config.json \
  --include adapter_model.safetensors \
  --include head.pt \
  --local-dir kev-4b/cuda/artifacts/kev
```

## Export

Selected mixed package:

From this recipe directory:

```bash
olive run --config kev-4b_cuda_mixed.json
```

Download the artifact under `artifacts/kev/` in this directory. Intermediate
packages are retained under `cache/mobius-export/`, so a rerun can reuse a
complete FP32 export after an interrupted mixed export. Olive atomically
publishes the selected package to its `build/` output.

## Output

The selected package is:

```text
build/
```

It contains the merged KEV adapter in an FP16 Qwen3.5 backbone with
middle-layer MLP projections quantized to symmetric INT4 block-128, the FP32
pointer head, base tokenizer files, `component_manifest.json`, and
`inference_model.json`. The INT4 projections use `MatMulNBits` accuracy level 3
for BF16 accumulation; this preserves the qualified development-suite accuracy
while reducing CUDA latency relative to level 4.
