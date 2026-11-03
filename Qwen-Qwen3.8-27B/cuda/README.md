# Qwen3.8-27B INT4 with DFlash2 speculative decoding (CUDA)

This recipe exports [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
as an ONNX Runtime GenAI package for the CUDA execution provider. It uses:

- symmetric INT4 target-model weights with block size 32;
- CPU token embeddings and a pruned language-model head;
- paged attention with a 256-token block size and INT8 per-channel KV cache;
- an INT4 DFlash2 drafter with seven draft tokens; and
- memory-selected runtime profiles for GPUs with 32 GiB or more of device memory.

The target and drafter use prepacked `MatMulNBits` weights. CUDA graphs, GEMM
auto-tuning, and 512-row `MatMulNBits` chunking are enabled in both decoder
sessions.

## Setup

Install compatible development builds of Olive and ONNX Runtime GenAI with
CUDA support. The recipe uses Model Builder runtime-profile and DFlash2
features that may not be available in the latest stable packages.

The target checkpoint is downloaded from Hugging Face. The recipe currently
sets the DFlash2 checkpoint path to `Qwen3.8-27B-DFlash2`; place that checkpoint
at the path resolved by Olive, or update `drafter_options.path` to its local or
Hugging Face location before exporting.

## Export

Run from this directory so that the included
`kv_scales_int8_per_channel.json` file is available to Model Builder:

```bash
olive run --config Qwen-Qwen3.8-27B_cuda_q4_int8kv_cpu_embed_dflash2_int4.json
```

The exported package is written under
`qwen_3.8_27b_int4_int8kv_cpu_embed_dflash2_int4/`. Its maximum configured
sequence length is 262,144 tokens.

## Runtime profiles

The generated package selects a runtime configuration from total GPU memory:

| Total GPU memory | KV blocks | Maximum batch size | Scheduled/chunk tokens |
|---|---:|---:|---:|
| 32 to less than 40 GiB | 1,031 | 1 | 1,024 |
| 40 to less than 60 GiB | 2,061 | 2 | 1,024 |
| 60 to less than 70 GiB | 3,091 | 3 | 1,024 |
| 70 GiB or more | 4,122 | 4 | 1,024 |

The base configuration uses 407 KV blocks, batch size 1, and a 512-token
schedule. It is the fallback when none of the profiles above is eligible.
Actual capacity also depends on runtime version, CUDA allocations, and other
processes using the GPU.

## Inference

Use ONNX Runtime GenAI's upstream
[`model-qa.py`](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py)
sample with the generated directory that contains `genai_config.json`:

```bash
python /path/to/onnxruntime-genai/examples/python/model-qa.py \
  --model_path /path/to/generated/model \
  --execution_provider cuda \
  --user_prompt "Explain why the sky appears blue." \
  --non_interactive \
  --timings
```

The package is text-only and requires CUDA; it does not include a vision
encoder or multimodal processing pipeline.
