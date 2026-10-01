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
  --local-dir artifacts/kev
```

## Export

```bash
python kev-4b/cpu/export.py \
  --artifact artifacts/kev \
  --output-dir build/kev
```

The package is written to:

```text
build/kev/kev-4b-fp32/
```
