# KEV-0.8B CUDA Mobius export

This recipe exports `jaredpalmer/kev-0.8b` as an FP16 multi-component package
for Foundry Local.

## Setup

From the `olive-recipes` repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r kev-0.8b/cuda/requirements.txt
```

Download the pinned adapter and head:

```bash
hf download jaredpalmer/kev-0.8b \
  --revision bf75a6a8848ea6960ff2ed108d9ed44c2941174f \
  --include adapter_config.json \
  --include adapter_model.safetensors \
  --include head.pt \
  --local-dir kev-0.8b/cuda/artifacts/kev
```

## Export

From this recipe directory:

```bash
olive run --config kev-0.8b_cuda_fp16.json
```

The package is staged under `cache/mobius-export/` and atomically published to
`build/`. The recipe JSON writes one intra-op and one inter-op host thread into
`genai_config.json`. INT4 is intentionally not enabled: the 4B quantization
policy has not received full KEV-0.8B accuracy qualification.
