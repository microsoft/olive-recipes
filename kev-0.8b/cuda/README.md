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

## Precision qualification

On one A100 80 GB, 100 measured in-process requests produced:

| Variant | Cache-hit p50 | Requests/s | GPU memory | Package size |
|---|---:|---:|---:|---:|
| FP32 | 9.07 ms | 110.26 | 4,547 MiB | 3.10 GB |
| **FP16** | **7.41 ms** | **134.84** | 2,633 MiB | 1.56 GB |
| INT4 middle MLP, edge 4 | 7.78 ms | 128.23 | 2,635 MiB | 1.30 GB |
| INT4 middle MLP, edge 8 | 7.68 ms | 130.16 | **2,509 MiB** | 1.43 GB |

FP16 is both faster than the INT4 variants and substantially closer to FP32.
Its sample probabilities differed by at most 0.0003.

Both INT4 policies were evaluated against matched FP32 on decision-v7,
devtools-v1, hard-v1, and transfer-v4: 3,568 records and 4,389 questions per
variant, with no rejection or truncation. Both worsened NLL and Brier score on
all four suites. The edge-4 Brier regression ranged from 0.0016 to 0.0135; the
edge-8 regression ranged from 0.0001 to 0.0100. INT4 is therefore rejected for
the supported recipe.
