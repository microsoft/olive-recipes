# CLM-v0.1-8B CUDA Mobius export

This recipe exports `Contrastive-LM/CLM-v0.1-8B` as a multi-component
non-generative package for Foundry Local.

## Setup

From the `olive-recipes` repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Contrastive-LM-CLM-v0.1-8B/cuda/requirements.txt
```

Download the pinned CLM head:

```bash
hf download Contrastive-LM/CLM-v0.1-8B \
  --revision e939398d4556fcd9400c76fa8c5a513202f42b0a \
  --include CLM_v0.1-8B.pt \
  --local-dir artifacts/clm
```

## Export

Selected mixed package:

From this recipe directory:

```bash
olive run --config CLM-v0.1-8B_cuda_mixed.json
```

Download the artifact under `artifacts/clm/` in this directory. A complete
FP32 intermediate is reused after an interrupted mixed export.

## Output

The selected package is:

```text
build/
```

It contains an FP16 Qwen3 backbone with middle-layer MLP projections quantized
to symmetric INT4 block-128, FP32 CLM heads, tokenizer files,
`component_manifest.json`, and `inference_model.json`.
