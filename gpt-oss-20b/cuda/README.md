# gpt-oss-20b — CUDA optimization

This folder contains Olive recipes for optimizing gpt-oss-20b targeting the CUDA EP.

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: INT4 precision by default

## Recipes

### Recommended variants

- `gpt-oss-20b_cuda_int4_mixed_int4_qmoe_int8_kv_windowed.json`
   - Targets 16 GB-class GPUs with INT4 block-64 QMoE experts, a mixed INT4/INT8 dense body, INT8 per-channel KV cache, and reduced caches for sliding-window layers.

- `gpt-oss-20b_cuda_int4_mixed_int4_qmoe_int8_kv_windowed_paged.json`
   - Uses the same INT4 mixed quantization with windowed PagedAttention, 256-token prefill chunks, and CUDA graphs for continuous batching through the GenAI `Engine` API.

- `gpt-oss-20b_cuda_int8_mxfp4_qmoe_int8_kv_windowed.json`
   - Targets 24 GB-class GPUs with MXFP4 block-32 QMoE experts, INT8 dense weights, INT8 per-channel KV cache, and reduced caches for sliding-window layers.

- `gpt-oss-20b_cuda_int8_mxfp4_qmoe_int8_kv_windowed_paged.json`
   - Uses the same INT8/MXFP4 quantization with windowed PagedAttention, 256-token prefill chunks, and CUDA graphs for continuous batching through the GenAI `Engine` API.

The non-paged recipes use `GroupQueryAttention` and can be run with the GenAI `Generator` API. The paged recipes require the `Engine` and `Request` APIs. Both recipe families keep the 12 local-attention layers in a reduced windowed KV cache and enable CUDA graph capture.

These recipes require a Model Builder version that supports the following combinations:

- `precision=int8` with `moe_quant_type=mxfp4` for the INT8-dense/MXFP4-expert layout.
- `windowed_kv_cache=true` with `use_paged_attention=true` and `paged_chunk_size` for the paged variants.

For the validated memory footprint, the generated `genai_config.json` must contain the string-valued decoder session option `"session.use_device_allocator_for_initializers": "1"`. Model Builder versions that do not emit it automatically can be fixed after the build:

```bash
python - model_int4_mixed_int4_qmoe_int8_kv_windowed \
   model_int4_mixed_int4_qmoe_int8_kv_windowed_paged \
   model_int8_mxfp4_qmoe_int8_kv_windowed \
   model_int8_mxfp4_qmoe_int8_kv_windowed_paged <<'PY'
import json
import sys
from pathlib import Path

for model_dir in map(Path, sys.argv[1:]):
   config_path = model_dir / "genai_config.json"
   config = json.loads(config_path.read_text())
   session_options = config["model"]["decoder"]["session_options"]
   session_options["session.use_device_allocator_for_initializers"] = "1"
   config_path.write_text(json.dumps(config, indent=4) + "\n")
PY
```

## Setup

1) Install the main branch of Olive:
   - pip install git+https://github.com/microsoft/olive.git
2) Install the appropriate runtime package for this backend:
   - onnxruntime-genai-cuda (CUDA build)
3) Run Olive to build/optimize the model
   - `olive run --config gpt-oss-20b_cuda_int4_mixed_int4_qmoe_int8_kv_windowed.json`

Additional notes:
- Requires NVIDIA GPU with CUDA support.
- Ensure CUDA toolkit and cuDNN are properly installed.
- The release-candidate recipes require an ONNX Runtime GenAI build with quantized KV cache and windowed PagedAttention model-builder support.

---

This README was auto-generated for the CUDA EP of gpt-oss-20b.
