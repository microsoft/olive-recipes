# Qwen3.8-27B INT4 (WebGPU)

This recipe exports [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
as an ONNX Runtime GenAI package for the WebGPU execution provider. It uses:

- symmetric INT4 target-model weights with block size 32;
- pruned language-model head;
- paged attention with a 256-token block size; and
- memory-selected runtime profile for GPUs with 40 to 60 GiB of device memory.

## Export

```bash
olive run --config Qwen-Qwen3.8-27B_webgpu_int4.json
```

The exported package is written under `qwen_3.8_27b_webgpu_int4_paged/`.

## Runtime profiles

The generated package selects a runtime configuration from total GPU memory:

| Total GPU memory | KV blocks | Maximum batch size | Scheduled/chunk tokens |
|---|---:|---:|---:|
| 40 to less than 60 GiB | 8 | 1 | 256 |

The base configuration uses 8 KV blocks, batch size 1, a 64-token schedule,
and chunk size 256. It is the fallback when the profile above is not eligible.
