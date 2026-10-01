# KEV-4B CPU Mobius export

This recipe exports `jaredpalmer/kev-4b` as an FP32 multi-component package
for Foundry Local.

## Setup

From the `olive-recipes` repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r kev-4b/cpu/requirements.txt
```

Download the pinned adapter and head:

```bash
hf download jaredpalmer/kev-4b \
  --revision 139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --include adapter_config.json \
  --include adapter_model.safetensors \
  --include head.pt \
  --local-dir kev-4b/cpu/artifacts/kev
```

## Export

From this recipe directory:

```bash
olive run --config kev-4b_cpu_fp32.json
```

Download the artifact under `artifacts/kev/` in this directory. The package is
staged under `cache/mobius-export/`, so a complete FP32 export is reused on a
rerun. Olive atomically publishes the selected package to its `build/` output.

```text
build/
```
