# Qwen3.8-27B CUDA optimization

This folder contains Olive recipes for Qwen3.8-27B targeting the CUDA EP with a per-channel
quantized paged KV cache and an INT4 DFlash 2 block drafter. The target can use either
blockwise INT4 weights or a prequantized mixed FP8/NVFP4 checkpoint, with INT4 or INT8 KV.

## What this folder is for

- Execution Provider: CUDA EP
- Precision: INT4 or mixed FP8/NVFP4 target weights, INT4 drafter weights, and an INT4 or
  INT8 per-channel quantized KV cache

Qwen3.8-27B is a hybrid model: 64 layers with `full_attention_interval=4`, so only the 16
full-attention layers (layer IDs 3, 7, ..., 63) carry a KV cache. The remaining 48 layers use
gated delta-net linear attention and are unaffected by these options.

## Recipe

- `Qwen-Qwen3.8-27B_cuda_int4_int4_per_channel_kv_paged_dflash2_int4.json`
  - INT4 block-32 target weights and INT4 per-channel paged KV.
  - INT4 block-32 DFlash 2 body and LM head with seven draft tokens.
  - Shared target/drafter embedding initializer.
- `Qwen-Qwen3.8-27B_cuda_int4_int4_per_channel_kv_paged_dflash2_int4_24gb.json`
  - The same INT4 target and INT4 DFlash 2 drafter, retuned to fit a 24 GB card.
  - Adds `op_types_to_quantize=MatMul/Gather`, which quantizes the embedding table to INT4.
  - Ships a single-stream runtime configuration rather than the concurrent-serving one.
  - The longest-context 24 GB option: 114,688 tokens, at -1.75 pp MMLU-Pro.
  - See [Fitting 24 GB](#fitting-24-gb).
- `Qwen-Qwen3.8-27B_cuda_int4_int8_embed_int4_per_channel_kv_paged_dflash2_int4_24gb.json`
  - The `_24gb` recipe with the embedding table at INT8 instead of INT4, via a per-node
    `quant_config` override on `/model/embed_tokens/Gather`.
  - Costs 606 MiB against the INT4 table, so `num_blocks` drops from 512 to 448 and the
    longest prompt from 114,688 to 98,304 tokens.
  - Scores as well as a dense FP16 table (83.000% vs 83.375%, *p* = 0.78), so this is the
    better default unless the extra 16K of context is needed.
- `Qwen-Qwen3.8-27B_cuda_nvfp4_int4_per_channel_kv_paged_dflash2_int4.json`
  - Preserves the target checkpoint's native FP8 projections and NVFP4 MLP weights.
  - Uses BF16 for unquantized target tensors and model I/O.
  - Uses the same INT4 per-channel paged KV and an INT4 DFlash 2 body. The drafter LM head
    stays dense because the target's native FP8 head is not a shareable MatMulNBits weight.
- `Qwen-Qwen3.8-27B_cuda_nvfp4_int8_per_channel_kv_paged_dflash2_int4.json`
  - Uses the same mixed FP8/NVFP4 target and INT4 DFlash 2 configuration.
  - Keeps the calibrated KV scales at their native INT8 `qmax=128` instead of rescaling them
    for INT4, providing the higher-precision KV-cache variant.

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

The NVFP4 recipe reads `unsloth/Qwen3.8-27B-NVFP4` by default. Model Builder detects its
compressed-tensors metadata and preserves the checkpoint's FP8/NVFP4 tensors instead of
requantizing them. To use a local checkpoint, override `input_model.model_path` with its path.

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

## Draft width

`max_draft_tokens` is a separate, runtime-side cap. `dflash2_num_draft_tokens` fixes what the
drafter *exports*; `max_draft_tokens` decides how many of those tokens the engine *verifies*
each step, and lands in `genai_config.json` as `speculative.max_draft_tokens`. Leaving it unset
applies the runtime default of 4, which wastes throughput: the drafter's block costs the same
to run whether 4 or 7 of its tokens are verified, so the first few extra tokens are nearly free.

Each recipe ships the measured optimum for its KV precision. Decode throughput on one H200
(tokens/s, six samples per point, min/max ranges non-overlapping between the winner and its
neighbours), for a 512-token single-stream prompt and for four concurrent streams:

| KV | k=4 (default) | k=5 | k=6 | k=7 |
| --- | --- | --- | --- | --- |
| INT8, single stream | 210.8 | 233.8 | **241.7** | 231.4 |
| INT8, four streams | 467.5 | 464.7 | **479.6** | 468.6 |
| INT4, single stream | 167.5 | 167.9 | 178.3 | **187.5** |
| INT4, four streams | 409.3 | 418.4 | **421.6** | 417.9 |

KV precision does move the optimum. INT4 KV is lossier, so it accepts fewer drafted tokens per
round than INT8 at the same width (2.34 vs 2.48 accepted at k=4), and its verification step is
also more expensive. It therefore has to propose wider to recover the same accepted-token count
and peaks at 7. INT8 KV saturates at 6, where the extra acceptance from a seventh token no
longer pays for the wider verification step. Hence `max_draft_tokens=6` in the INT8 recipe and
`7` in the INT4 ones. Only the two NVFP4 recipes were measured; the INT4-weight recipe inherits
the INT4-KV setting.

This option changes *how* tokens are produced, not *which*: speculative decoding verifies every
drafted token against the target, and the generated token sequences were identical across every
value of `k` above. Retuning it needs no re-export, only a config edit.

## Prefill chunking, and the single-stream optimum

The recipes ship a concurrent-serving configuration: `max_batch_size=16`, `paged_chunk_size=256`
and `max_scheduled_tokens=1024`. That is not the fastest setting for one request at a time, and
the difference is large enough to be worth calling out.

`max_scheduled_tokens` caps the tokens in one dynamically batched forward pass; `paged_chunk_size`
becomes `search.chunk_size`, which caps how many prompt tokens a *single* request contributes to
that pass. With several requests competing, the small per-request cap is what lets their prefills
interleave instead of one request consuming the whole budget. At `max_batch_size=1` there is
nobody to interleave with, so the two knobs collapse into one — the effective chunk is just their
minimum — and a small chunk becomes a pure loss, because a 256-token forward pass cannot saturate
the GPU.

Measured on one H200 with the NVFP4 target and INT4 per-channel KV, single stream, a 16,384-token
prompt, three timed repetitions after a warmup. Peak memory is device-level usage from an idle
baseline:

| chunk = `max_scheduled_tokens` | TTFT (s) | Prefill tok/s | Decode tok/s | Peak MiB |
| ---: | ---: | ---: | ---: | ---: |
| 256 (recipe default) | 4.71 | 3,480 | 50.8 | 30,828 |
| 1,024 | 3.01 | 5,451 | 50.3 | 31,486 |
| 2,048 | 2.72 | 6,032 | 50.5 | — |
| 4,096 | 2.58 | 6,357 | 49.9 | 36,542 |
| 8,192 | 2.56 | 6,410 | 50.6 | 42,686 |

Sweeping `paged_chunk_size` with a large `max_scheduled_tokens` and sweeping `max_scheduled_tokens`
with a large `paged_chunk_size` produced the same curve to within 0.5%, which confirms that only
the minimum of the two matters at this batch size. Decode is flat throughout: this knob buys
prefill and nothing else.

4,096 is the knee. It is worth **1.8x prefill throughput** over the shipped 256 for 5.7 GiB of
extra peak memory, while 8,192 adds a further 6.1 GiB for under 1%. The memory is the prefill
logits transient, `chunk x 248320 x 2` bytes, since these recipes do not set `prune_lm_head`.

For a latency-sensitive single-stream deployment, override these four fields in the exported
`genai_config.json` — all of them are runtime values, so **no re-export is needed**:

```json
{
  "search": { "chunk_size": 4096 },
  "engine": { "dynamic_batching": { "max_batch_size": 1, "max_scheduled_tokens": 4096 } }
}
```

Do not carry this back into a serving deployment: at eight concurrent 2,048-token prompts the
shipped `chunk_size=256` was worth +25% aggregate throughput and -25% worst-case TTFT against
`1024`, for exactly the interleaving reason above.

`paged_block_size` and `max_draft_tokens` do **not** change at `max_batch_size=1`; the shipped 256
and 7 are already the single-stream optimum. See below for the block-size evidence.

## Paged block size

`paged_block_size=256` is baked into the exported KV-cache shapes, so unlike the knobs above it
needs a re-export to change. ONNX Runtime's PagedAttention accepts any power of two of at least
16; the recipes use 256.

Smaller pages are often suggested for cutting KV-pool fragmentation, but on this model they do not
pay. Measured on one H200 with the NVFP4 target and INT4 per-channel KV, one candidate exported
per block size, with
`num_blocks` scaled inversely so every configuration holds the same 262,144-token pool, and with
the drafter removed so the numbers describe the target alone:

| `paged_block_size` | Prefill tok/s (1K / 4K / 16K prompt) | Decode tok/s (1K / 4K / 16K) |
| ---: | ---: | ---: |
| 32 | 3,995 / 3,960 / 3,714 | 63.9 / 48.2 / 22.8 |
| 64 | 3,989 / 3,966 / 3,733 | 64.1 / 48.2 / 23.0 |
| 128 | 3,986 / 3,960 / 3,710 | 70.9 / 72.1 / 68.4 |
| 256 (recipe default) | 4,022 / 3,953 / 3,717 | 72.7 / 71.4 / 69.4 |
| 512 | 4,014 / 3,950 / 3,735 | 72.2 / 71.2 / 70.2 |

Prefill is flat: under 1% across a 16x range of block sizes, at every prompt length. The whole
cost of a small page lands on decode, and it grows with context — 12% at 1K, 33% at 4K and 67% at
16K — because a sequence spanning more pages costs more block-table indirection in the paged decode
kernel. 128, 256 and 512 are equivalent within noise.

One further constraint applies to the drafter. ONNX Runtime prefers its FlashAttention paged
kernel, which requires the page to be a multiple of its tile: 128 for a head size of 128, 64 above
that. The target here has `head_size=256`, but the DFlash 2 drafter has `head_size=128`, so the
drafter is what puts a floor of 128 on this recipe. On ONNX Runtime releases before non-causal
PagedAttention gained its non-FlashAttention backends, a page below that floor does not merely slow
the drafter down, it fails its attention node outright and the engine silently continues with
target-only decoding:

```
PagedAttention: is_causal=0 requires the FlashAttention backend
(sm>=80, fp16/bf16, head_size 128, block_size 64).
```

## Fitting 24 GB

The `_24gb` recipe targets a 24 GB consumer card (RTX 4090, 24,564 MiB total, so a working
budget of 23,500 MiB). The default recipe does not fit: its weights alone occupy 19,372 MiB of
device memory, which leaves too little for the KV pool to hold a useful context.

Two export changes cut the weight side, measured as unique bytes across both graphs on one
H200:

| | `model.onnx.data` | `dflash2.onnx.data` | unique total |
| --- | ---: | ---: | ---: |
| default recipe | 16,272 | 2,045 | 18,317 |
| + drafter adopts the target's LM head | 16,272 | 1,363 | 17,635 |
| + `op_types_to_quantize=MatMul/Gather`, INT8 table | 15,136 | 1,363 | 16,498 |
| + `op_types_to_quantize=MatMul/Gather`, INT4 table | 14,529 | 1,363 | **15,892** |

The LM-head saving is automatic and has no option. The embedding saving is `Gather`, and it
only pays if the drafter adopts the quantized table too: the two graphs share
`model.embed_tokens.weight` by name, so a target that renames it to
`model.embed_tokens.weight_Q4` (or `_Q8`) while the drafter keeps a dense `Gather` breaks the
sharing and costs *more* than the target saved. Model Builder handles this; the point is that
the drafter and the target must agree on the embedding format, exactly as they must on the LM
head.

On device that is 19,372 -> 16,718 MiB of graphs, and the freed 2,654 MiB goes to the KV pool.
Measured peak and the longest prompt accepted, single stream, `chunk_size=512`,
`max_scheduled_tokens=512`, `max_draft_tokens=7`:

| recipe | `paged_block_size` | `num_blocks` | peak MiB | longest prompt |
| --- | ---: | ---: | ---: | ---: |
| default knobs | 128 | 256 | 22,446 | 28,672 |
| `_24gb` | 256 | 384 | 21,228 | 65,536 |
| `_24gb` | 256 | **512** | **22,764** | **114,688** |
| `_24gb` | 256 | 640 | 23,788 | 147,456 |
| `_int8_embed_..._24gb` | 256 | 384 | 22,250 | 65,536 |
| `_int8_embed_..._24gb` | 256 | **448** | **22,762** | **98,304** |
| `_int8_embed_..._24gb` | 256 | 512 | 23,788 | 114,688 |

512 is the shipped value for `_24gb`: **4x the context of the default knobs at the same peak**,
with 740 MiB of headroom against the 23,500 MiB budget. 640 fits 147,456 tokens but peaks 288
MiB over budget, so it is only safe on a card with nothing else resident. The INT8-table recipe
ships 448, which lands on the same peak as `_24gb` at 512 — the 606 MiB the wider table costs
is very nearly one `num_blocks` step of 64.

Two measurement traps are worth repeating. `num_blocks` is a **byte budget**, not a block
count: the engine splits it between the target pool and the drafter's auxiliary ring, so usable
context runs ~12% below `num_blocks * paged_block_size`. And the pool allocates lazily in ~1 GiB
granules, so peak memory must be sampled during a full-length request — a short prompt
under-reports it by up to a granule.

### The embedding precision is the real knob

`op_types_to_quantize=MatMul/Gather` is the one change here that is not free. Three builds
differing in *only* how the embedding table is quantized, MMLU-Pro on the 800-sample
stratified subset, thinking mode, greedy, all four shards matched:

| embedding | MMLU-Pro | vs FP16 | McNemar p |
| --- | ---: | ---: | ---: |
| FP16 `Gather` | **83.375%** (667/800) | - | - |
| INT8 `GatherBlockQuantized` | **83.000%** (664/800) | -0.375 pp | 0.775 |
| INT4 `GatherBlockQuantized` | **81.625%** (653/800) | -1.750 pp | 0.065 |

INT8 is indistinguishable from the dense table. INT4 is the only arm that separates, and its
direction is consistent with draft acceptance, which is equal to 4K context but drops from
0.999 to 0.929 at 16K.

Calibrate those *p* values against the harness's own noise: the same INT8 model re-run with 8
shards instead of 4 scored 82.250% rather than 83.000%, a 0.75 pp swing from batch composition
alone. So the INT8-vs-INT4 gap (+1.375 pp) is about twice the noise floor and the
INT8-vs-FP16 gap is inside it.

Now the memory side, measured the same way as the table above:

| embedding | unique weight MiB | engine MiB | `num_blocks` | peak MiB | longest prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 17,635 | 24,444 | any | >=25,322 | **does not fit** |
| INT8 | 16,498 | 21,372 | **448** | 22,762 | 98,304 |
| INT4 | 15,892 | 21,372 | **512** | 22,764 | 114,688 |

The FP16 row is an accuracy reference, not an option: its engine footprint is 24,444 MiB
before the KV pool allocates anything, which exceeds a 4090's 24,564 MiB total at *any*
`num_blocks`. Dropping `Gather` from `op_types_to_quantize` does not buy accuracy on this
card; it buys a model that will not load.

That leaves INT8 against INT4, at essentially the same peak: **1.375 pp of MMLU-Pro against
16,384 tokens of context**. `_24gb` ships INT4 for the longest context;
`_int8_embed_..._24gb` ships INT8 and is the better default if 98K tokens is enough, because
it gives up nothing measurable against a dense table.

INT8 is reached with a per-node override rather than a new flat option:

```json
"op_types_to_quantize": "MatMul/Gather",
"quant_config": "[{\"match\": {\"name\": \"/model/embed_tokens/Gather\"}, \"type\": \"int8\"}]"
```

### What is *not* in this recipe

`block_size=64` for the target's INT4 weights would save a further 745 MiB, but it does not
load on a default ONNX Runtime build. `CheckFpAIntBEligibility` accepts prepacked MatMulNBits
weights at `block_size=32` only when `USE_COMPACT_FPA_INTB_GEMM` is set, and the compact kernel
set is the default (`onnxruntime_USE_FPA_INTB_GEMM_FULL` is off):

```
This compact fpA_intB build supports prepacked weights only for FP16 activations,
INT4 or INT8 weights, block_size=32, ... Got bits=4, block_size=64, weight_prepacked=1
```

Unlocking it needs either a full-kernel-set ONNX Runtime build, or dropping
`matmulnbits_weights_prepacked`, which gives up the fpA_intB GEMM and its decode throughput.
Neither is a good default, so the recipe stays at `block_size=32`.

## CUDA graphs

`enable_cuda_graph=true` writes `enable_cuda_graph=1` into the decoder's CUDA provider options,
and the engine then captures a graph per distinct decode shape. It needs a runtime whose paged
decode path supports capture; without that support the flag is inert rather than harmful.
Measured on the same H200 configuration:

| Case | Eager | Graphs | Speedup |
| --- | --- | --- | --- |
| Single stream, 512-token prompt | 199.1 | 232.0 | 1.17x |
| Single stream, 2048-token prompt | 148.2 | 170.7 | 1.15x |
| Four concurrent streams | 442.2 | 468.0 | 1.06x |

Peak device memory rises 2.8% (41.0 to 42.2 GiB) for the captured graph pool. The first few
steps at each new shape pay the capture cost, so short runs see less benefit than the
steady-state numbers above.

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
  Dropping to 1 also makes the shipped prefill chunk the wrong choice; see
  [Prefill chunking, and the single-stream optimum](#prefill-chunking-and-the-single-stream-optimum).

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
