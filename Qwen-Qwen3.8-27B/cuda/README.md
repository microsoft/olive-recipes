# Qwen3.8-27B CUDA optimization

This folder contains an Olive recipe for Qwen3.8-27B targeting the CUDA EP with INT4 weights,
an INT4 per-channel paged KV cache, and an INT4 DFlash 2 block drafter.

## What this folder is for

- Execution Provider: CUDA EP
- Precision: INT4 target/drafter weights with an INT4 per-channel quantized KV cache

Qwen3.8-27B is a hybrid model: 64 layers with `full_attention_interval=4`, so only the 16
full-attention layers (layer IDs 3, 7, ..., 63) carry a KV cache. The remaining 48 layers use
gated delta-net linear attention and are unaffected by these options.

## Recipe

- `Qwen-Qwen3.8-27B_cuda_int4_int4_per_channel_kv_paged_dflash2_int4.json`
  - INT4 block-32 target weights and INT4 per-channel paged KV.
  - INT4 block-32 DFlash 2 body and LM head with seven draft tokens.
  - Shared target/drafter embedding initializer.

The recipe uses PagedAttention with 256-token blocks and CUDA graphs, so it must be driven
through the ONNX Runtime GenAI `Engine` and `Request` APIs, not the `Generator` API. The
drafter recipe is paged-only by construction: `dflash2_path` and `state_update_capacity` both
require `use_paged_attention=true`.

## DFlash 2 drafter

The recipe needs the draft checkpoint on disk. `dflash2_path` is opened directly as a local
directory, so it is not resolved as a Hugging Face repo id; download it first:

```bash
hf download incoai/Qwen3.8-27B-DFlash2 --local-dir Qwen3.8-27B-DFlash2
```

The recipe uses the relative path `Qwen3.8-27B-DFlash2`, so run Olive from this folder or edit
`dflash2_path` to an absolute path.

`aux_hidden_state_layers` must be each `target_layer_ids` entry **plus one**: the drafter taps
layer outputs while the option names the residual stream entering a layer. This checkpoint has
`target_layer_ids = [5, 19, 33, 47, 61]`, hence `6,20,34,48,62`. A mismatch is rejected by the
builder with the expected list in the message.

`dflash2_num_draft_tokens` is capped at the checkpoint's block size minus the anchor token,
so 7 here (`block_size = 8`), which is also the default. `state_update_capacity=7` matches it
so the recurrent state transitions for a whole draft block are captured in one forward.

Exporting a block drafter forces `search.do_sample=false` in `genai_config.json`, and the
builder prints when it does. This is deliberate: the engine only requests drafts for greedy
turns, so a sampled configuration would silently produce zero drafts. `top_k`, `top_p` and
`temperature` are left untouched.

The recipe sets `dflash2_precision=int4`. Its quantized body uses the plain blockwise layout
because the prepacked fpA-intB kernel accepts FP16 activations while the drafter body uses
bf16. Its FP16 LM head remains eligible for the target's prepacked layout.

Initializer sharing between the two graphs is automatic and has no option, but it folds by
initializer *name and bytes*. The verified artifact shares `model.embed_tokens.weight`, saving
2,543 MB. The independently quantized target and drafter LM-head bytes did not match, so the
exporter correctly retained a separate quantized copy.

## Calibration scales

Per-channel KV quantization is static, so `kv_cache_scale_file` is required. The recipe reads
the committed `kv_scales_int8_per_channel.json`:

```json
{
  "scales": {"k_scales": [[...1024 floats...], ...16 layers...],
             "v_scales": [[...1024 floats...], ...16 layers...]},
  "layer_ids": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63],
  "qmax": 128.0
}
```

Each per-layer vector is `num_key_value_heads * head_size` = 4 × 256 = 1024 entries, and
`layer_ids` maps the 16 entries onto the model's full-attention layers.

The calibrated scale is `threshold / qmax`,
and qmax is 128 for INT8 but 8 for INT4. Because the file declares `"qmax": 128.0`, the builder
rescales it by 128/8 when the requested scheme is `int4_per_channel`. A scale file **without**
`qmax` is interpreted as already matching the target scheme, so reusing an INT8 file for INT4
would under-scale by 16× and clip the cache. Keep the key if you regenerate the file.

To calibrate your own, run the model over a representative corpus, record a per-channel
absolute-magnitude threshold for each full-attention layer's K and V, divide by the qmax you
calibrated at, and record that qmax in the file.

## Setup

1) Install the main branch of Olive:
   - `pip install git+https://github.com/microsoft/olive.git`
2) Install the runtime package for this backend:
   - `onnxruntime-genai-cuda`
3) Run Olive to build/optimize the model
  - `olive run --config Qwen-Qwen3.8-27B_cuda_int4_int4_per_channel_kv_paged_dflash2_int4.json`

Additional notes:

- Requires an NVIDIA GPU with CUDA support, plus a matching CUDA toolkit and cuDNN.
- INT4 paged KV needs an ONNX Runtime built with `onnxruntime_USE_INT4_KV_CACHE=ON`. This is
  the default and ships in the `onnxruntime-gpu==1.30.0` release wheel, so a custom build is
  not required.
- `matmulnbits_weights_prepacked=1` selects the SM80 prepack layout, which accepts block size
  32 and runs on SM75–SM120. Do **not** use `2` here: the SM90 layout only accepts block size
  64/128, so on this block-32 model it silently prepacks nothing while still advertising
  `ep.cuda.fpa_intb_gemm=1`.
- `use_device_allocator_for_initializers=true` keeps the prepacked weights from being
  duplicated in the CUDA BFC arena; it is worth roughly 15 GiB of device memory at load.
- `max_batch_size` is set to 16 so the `Engine` can serve concurrent requests out of the box.
  A model built with `max_batch_size=1` fails any run at concurrency > 1 with
  `EngineEventFlags.CAPACITY_BLOCKED`. Raise or lower it, or override
  `engine.dynamic_batching.max_batch_size` in `genai_config.json`, to match your deployment.

## Validation results

The recipe was exported from `Qwen/Qwen3.8-27B` and `incoai/Qwen3.8-27B-DFlash2` on an H200.
The target and drafter external payloads are byte-identical to a direct Model Builder reference.
The target graph has 401 prepacked `MatMulNBits` nodes. The DFlash2 graph has 47
`MatMulNBits` nodes, and only its FP16 LM head is prepacked, as required for the bf16 body.

### Latency and memory

These runs used ORT 1.30, an otherwise idle H200, paged attention, and 2,048 generated tokens
per request. Peak memory is device-level usage measured from an idle baseline.

| Prompt | Batch | Drafts | TTFT (ms) | ms / target forward | Decode tok/s | Peak MiB | Acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 7 | 368.66 | 25.594 | 325.66 | 26,540.875 | 99.72% |
| 8,192 | 4 | 7 | 2,360.90 | 42.806 | 378.28 | 28,278.875 | 94.52% |
| 32,768 | 4 | 7 | 8,812.43 | 123.894 | 209.97 | 29,302.875 | 100.00% |
| 32,768 | 4 | 0 | 8,520.84 | 39.095 | 107.00 | 25,516.938 | n/a |

All rows passed the timing eligibility gate. DFlash2 reported zero failures, disables,
admission misses, and fallback steps. At 32K/batch 4 it improved decode throughput by 1.96x
at a 3,785.94 MiB peak-memory cost. A speculative target forward verifies multiple tokens, so
`ms / target forward` is expected to be higher than the no-drafter row.

### MMLU-Pro and GPQA Diamond

Quality used greedy decoding, 7 draft tokens, an 8,192-token generation limit, and four fixed
sample shards. Each shard evaluated the direct Model Builder reference, an exact same-process
repeat, a concurrency-4 neutral control, and the Olive artifact in one process. Both exact
repeats and both Olive comparisons produced zero changed generations.

| Task | Arm | Correct | Accuracy | Changed generations | Delta vs. reference | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| MMLU-Pro (800) | Direct reference | 663/800 | 82.875% | - | - | - |
| MMLU-Pro (800) | Reference repeat | 663/800 | 82.875% | 0 | 0.000 pp | 1.000 |
| MMLU-Pro (800) | Concurrency-4 control | 659/800 | 82.375% | 602 | -0.500 pp | 0.652 |
| MMLU-Pro (800) | Olive recipe | 663/800 | 82.875% | 0 | 0.000 pp | 1.000 |
| GPQA Diamond (198) | Direct reference | 146/198 | 73.737% | - | - | - |
| GPQA Diamond (198) | Reference repeat | 146/198 | 73.737% | 0 | 0.000 pp | 1.000 |
| GPQA Diamond (198) | Concurrency-4 control | 149/198 | 75.253% | 183 | +1.515 pp | 0.720 |
| GPQA Diamond (198) | Olive recipe | 146/198 | 73.737% | 0 | 0.000 pp | 1.000 |

Across the Olive quality runs, MMLU-Pro completed 122,550 speculative rounds at 59.81%
acceptance and GPQA completed 123,638 rounds at 54.64% acceptance. Both tasks reported zero
DFlash2 failures, disables, admission misses, and fallback steps.
