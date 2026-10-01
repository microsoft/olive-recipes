# CLM-v0.1-8B CPU Mobius export

This recipe exports `Contrastive-LM/CLM-v0.1-8B` as an FP32
multi-component package for Foundry Local.

## Setup

From the `olive-recipes` repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Contrastive-LM-CLM-v0.1-8B/cpu/requirements.txt
```

Download the pinned CLM head:

```bash
hf download Contrastive-LM/CLM-v0.1-8B \
  --revision e939398d4556fcd9400c76fa8c5a513202f42b0a \
  --include CLM_v0.1-8B.pt \
  --local-dir artifacts/clm
```

## Export

```bash
python Contrastive-LM-CLM-v0.1-8B/cpu/export.py \
  --artifact artifacts/clm \
  --output-dir build/clm
```

The package is written to:

```text
build/clm/clm-v0.1-8b-fp32/
```
