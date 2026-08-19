# Qwen3.6-35B-A3B (Olive MoE quantization + Mobius + ONNX Runtime GenAI, CUDA)

Olive recipes that export [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
— a hybrid MoE model (40 layers, 256 experts, 8 experts/token, GatedDeltaNet
linear attention interleaved with full attention, plus a Qwen3-VL vision tower)
— to ONNX for ONNX Runtime GenAI on the CUDA execution provider.

Unlike the `NvTensorRtRtx` recipe in this repo (which uses ORT GenAI's
`ModelBuilder` + `MatMulNBitsToQDQ`), these recipes use Olive's MoE-aware
**PyTorch-side** weight-only quantization (`Rtn` / `KQuant`) followed by the
[`MobiusBuilder`](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
export pass (via [mobius](https://github.com/onnxruntime/mobius)), which
natively understands per-expert MoE weight layouts, hybrid linear/full-attention
graphs, and multi-component vision-language packages.

## Recipes

| Recipe | Loaded HF class | Quantizer | Exported components | Status |
|---|---|---|---|---|
| `rtn_fp16/config.json` | `Qwen3_5MoeForCausalLM` (text-only, default `text-generation` task) | `Rtn` int4, `group_size=128`, `sym`, `moe` | `decoder` (single `model.onnx`) | **Previously validated** (A100-80GB); re-validation pending with the pinned post-#2630 dependencies |
| `kquant_fp16/config.json` | `Qwen3_5MoeForCausalLM` (text-only, default `text-generation` task) | `KQuant` int4, `group_size=128`, `sym`, `moe` | `decoder` (single `model.onnx`) | **Not yet validated** — config-only; no full GPU run recorded |
| `vl_kquant_fp16/config.json` | `Qwen3_5MoeForConditionalGeneration` (multimodal, `task: image-text-to-text`) | `KQuant` int4, `group_size=128`, `sym`, `moe` | `decoder` + `vision_encoder` + `embedding` | **Blocked / not yet validated** — requires [mobius#515](https://github.com/onnxruntime/mobius/pull/515) |

All three target the CUDA EP and export at `fp16` (see
[Why `fp16` and why the EP matters](#why-precision-fp16-not-bf16-and-why-targetaccelerators-matter)).

`Qwen/Qwen3.6-35B-A3B` ships a **single multimodal checkpoint**
(`model_type: qwen3_5_moe`, `architectures: ["Qwen3_5MoeForConditionalGeneration"]`,
with the decoder hyper-parameters under `text_config` and a `vision_config`
sibling). The text-only recipes simply load it through the causal-LM class,
which materializes only the decoder; the VL recipe loads the conditional-generation
class, which materializes the decoder **and** the vision tower.

## Pipeline

Both text-only recipes are two passes:

1. **`Rtn` / `KQuant` pass** (`olive/passes/pytorch/rtn.py`,
   `olive/passes/pytorch/kquant.py`): 4-bit symmetric weight-only quantization,
   `group_size=128`, with `moe=true` so each of the 256 experts per layer is
   quantized independently (rather than merging all experts into a single
   blockwise-quantized tensor). `KQuant` uses llama.cpp's iterative
   weighted-least-squares k-quant search instead of plain round-to-nearest;
   it is slower than `Rtn` but usually lower error at the same bit width.
2. **`MobiusBuilder` pass**: exports the quantized PyTorch checkpoint straight
   to a full ORT GenAI package (`model.onnx` + `model.onnx.data` +
   `genai_config.json` + tokenizer files) — no intermediate ONNX conversion
   pass is needed.

The multimodal recipe is *the same two passes*, with `task: image-text-to-text`
on the input model. Nothing else changes.

## Multimodal: why one workflow is enough

The VL recipe deliberately does **not** use `MobiusBuilder`'s
`components_to_export` filter, and does not split the model into per-component
workflows that are later assembled by hand. That is unnecessary here:

- **Olive quantizes only what should be quantized, in one pass.** Since
  [Olive#2630](https://github.com/microsoft/Olive/pull/2630), Olive's
  `ModelWrapper` resolves the VL checkpoint's composite config
  (`text_config.*` fall-backs for `hidden_size` / `num_hidden_layers` /
  `num_attention_heads` / `num_key_value_heads` / `head_dim`) and the nested
  decoder path (`model.language_model.*`, next to `model.visual`). The
  quantization walk (`iter_quant_targets`) then covers the decoder only:

  | Module group | int4 quantized? | Why |
  |---|---|---|
  | `model.language_model.layers.*.self_attn.*`, `mlp.shared_expert.*` | yes | ordinary decoder `nn.Linear`s |
  | `model.language_model.layers.*.mlp.experts.{gate_up,down}_proj` (fused 3D) | yes (`moe: true`) | per-expert quantization, QMoE-compatible layout |
  | `model.visual.*` (vision tower) | **no** | vision towers are skipped unless `quantize_vision=true` |
  | `model.language_model.embed_tokens`, `lm_head` | **no** | `embeds` / `lm_head` default to `false` |
  | `mlp.gate` (router), `mlp.shared_expert_gate` | **no** | routing signals, excluded by Olive#2610 / Olive#2630 |
  | `linear_attn.*` (GatedDeltaNet projections) | **no** | SSM-style recurrent block, excluded by Olive#2630 (`MAMBA` → `linear_attn`) |

  So the vision encoder and the embedding table stay float automatically —
  no `modules_to_not_convert` gymnastics are required for this config.

- **Mobius builds all three components from that one checkpoint.** For a
  Qwen3.5/3.6-MoE checkpoint carrying a `vision_config`, mobius dispatches to
  `Qwen35MoEVL3ModelCausalLMModel`, which emits a 3-model ORT GenAI package:
  `decoder` (hybrid linear/full attention + QMoE FFN, consuming
  `inputs_embeds`), `vision_encoder` (Qwen3-VL ViT), and `embedding` (token
  embedding + image-feature fusion). Its `preprocess_weights` routes the HF
  keys (`model.visual.*`, `model.language_model.*`, `lm_head.*`) to the three
  sub-models and packs the Olive-format expert tensors into fused QMoE
  parameters. Olive's `MobiusBuilder` then returns a `CompositeModelHandler`
  with `no_flatten: True`, preserving the `<component>/model.onnx` layout that
  ORT GenAI expects, and writes the shared sidecars (`genai_config.json`,
  tokenizer files, `processor_config.json`) at the package root.

Splitting the export would mean re-assembling `genai_config.json` by hand —
`MobiusBuilder` explicitly skips GenAI config generation when
`components_to_export` is set, precisely because a partial package cannot
describe the full pipeline. The partial-export / per-component machinery —
Olive#2456 (`components_to_export` on `MobiusBuilder`), #2457
(`components_to_skip` on `OnnxBlockWiseRtnQuantization`) and #2480
(multi-build support) — is therefore **not** a dependency of this recipe.

### Required mixed-precision fix: mobius#515

With current Olive module selection, all three recipes require mobius#515.
Olive leaves
`linear_attn.*` and `shared_expert_gate` in floating point (see the table
above), but mobius builds those modules with the quantized-linear factory for
any Olive-format checkpoint — so graph construction expects packed
`MatMulNBits` initializers that the checkpoint does not contain, and weight
binding fails.

[mobius#515 — *Fix Qwen3.6 Olive mixed-precision export*](https://github.com/onnxruntime/mobius/pull/515)
aligns mobius's Qwen3.5/3.6-MoE graph construction with Olive's module
selection: for `quant_method == "olive"` MoE checkpoints, `linear_attn` and
`shared_expert_gate` are built as plain float `Linear`, while attention,
the shared-expert MLP, and the fused QMoE experts stay quantized. Dense,
non-Olive and unquantized graph construction is unchanged.

The pinned mobius revision in `requirements.txt` contains #515. The original
`rtn_fp16` validation predates Olive#2630, so it must also be re-run with these
dependencies. Treat both KQuant configs as unvalidated until their full GPU
runs complete.

## Why `precision: fp16` (not `bf16`) and why `target`/`accelerators` matter

Footguns hit while building these recipes, all easy to reproduce with a naive
config:

- **EP must be set explicitly via `systems`/`target`.** `MobiusBuilder` reads
  the execution provider from the Olive accelerator spec
  (`self.accelerator_spec.execution_provider`), not from a pass parameter. If
  `systems`/`target` are omitted, Olive defaults to `cpu-cpu`, and the CPU EP
  only fuses `GroupQueryAttention` for `fp32` — `fp16`/`bf16` on CPU silently
  fall back to the standard (non-GQA) `Attention` op, which is incompatible
  with the `past_present_share_buffer=True` mode required by this model's
  `LinearAttention` (recurrent-state) layers. Mobius raises a clear
  `ValueError` in that case rather than emitting a broken `genai_config.json`
  — the fix is to point the workflow at the CUDA EP (see the configs) and
  export with a precision that EP supports (CUDA's `gqa_dtypes` include both
  `FLOAT16` and `BFLOAT16`).
- **Use `fp16`, not `bf16`, on CUDA.** Even after fixing the EP, `bf16`
  export fails at ORT GenAI model-load time with:
  `Provider type for CausalConvWithState node ... is not set`. This is
  because ONNX Runtime's CUDA kernel for
  `com.microsoft.CausalConvWithState` (used by the GatedDeltaNet
  `LinearAttention` layers) is only registered for `float` and `MLFloat16` —
  there is no `BFloat16` CUDA kernel registration
  (`onnxruntime/contrib_ops/cuda/bert/causal_conv_with_state.cc`). `fp16`
  satisfies both the GQA-fusion dtype requirement and the
  `CausalConvWithState` kernel's registered types, so use `fp16` for CUDA
  exports of this architecture until/unless a `BFloat16` CUDA kernel is
  added upstream.
- **QMoE scales must match the export precision (fixed upstream).** Mobius's
  MoE component used to pin the `fc1_scales`/`fc2_scales` initializers to
  `FLOAT32` regardless of the requested export precision. ONNX Runtime's
  `com.microsoft::QMoE` kernel (`quant_type="int"`) requires those scales'
  dtype (`T2`) to exactly match the activation dtype (`T`) — there is no
  registered kernel for `T2=FLOAT32`/`T=FLOAT16`. With the mismatch, ORT
  silently found *no* matching QMoE kernel on either EP and fell back to
  running **every QMoE node — i.e. the entire MoE FFN compute — on the CPU
  EP**, even though the workflow explicitly targeted CUDA. This also broke
  `enable_cuda_graph`, since ORT GenAI's CUDA graph capture requires every
  decoder node to be assigned to the same EP. This is fixed in
  [mobius#505](https://github.com/onnxruntime/mobius/pull/505) (scales are
  now downcast to the model's target precision like every other parameter);
  make sure your `mobius` install includes that fix. With the fix, all QMoE
  nodes are correctly assigned to CUDA and `enable_cuda_graph: "1"` works
  out of the box with no manual `genai_config.json` edits.

## Setup

```bash
pip install -r cuda/requirements.txt
```

The requirements pin unreleased Olive and mobius revisions containing the
MoE-aware KQuant path, VL model-wrapper support, and mobius#515. The model
configs also pin the Hugging Face checkpoint revision and disable remote code.
Replace these pins with compatible release versions once those changes ship.

## Run

```bash
# text-only, RTN (validated)
olive run --config cuda/rtn_fp16/config.json

# text-only, KQuant
olive run --config cuda/kquant_fp16/config.json

# multimodal (decoder + vision_encoder + embedding), KQuant
olive run --config cuda/vl_kquant_fp16/config.json
```

Paths in the configs (`output_dir`) are relative to the model recipe directory,
so run these from `Qwen-Qwen3.6-35B-A3B/`.

## Output layout

Text-only recipes produce a single-component ORT GenAI package:

```
cuda/rtn_fp16/models/          # or cuda/kquant_fp16/models/
├── model.onnx                 # ONNX graph
├── model.onnx.data            # External weight data (~20GB for this model)
├── genai_config.json          # ORT GenAI runtime configuration
├── tokenizer.json
├── tokenizer_config.json
└── chat_template.jinja
```

The multimodal recipe produces a 3-component package — each component in its
own sub-directory, with the shared sidecars at the package root:

```
cuda/vl_kquant_fp16/models/
├── decoder/
│   ├── model.onnx
│   └── model.onnx.data
├── vision_encoder/
│   ├── model.onnx
│   └── model.onnx.data
├── embedding/
│   ├── model.onnx
│   └── model.onnx.data
├── genai_config.json          # describes all three components
├── processor_config.json
├── tokenizer.json
├── tokenizer_config.json
└── chat_template.jinja
```

## Verify with `onnxruntime_genai` (text-only)

`inference.py` in this folder runs the text-only packages:

```bash
python cuda/inference.py --model-path cuda/rtn_fp16/models --prompt "The quick brown fox"
```

Equivalent minimal script:

```python
import onnxruntime_genai as og

model = og.Model("cuda/rtn_fp16/models")
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

params = og.GeneratorParams(model)
params.set_search_options(max_length=32, do_sample=False)

generator = og.Generator(model, params)
generator.append_tokens(tokenizer.encode("The quick brown fox"))
while not generator.is_done():
    generator.generate_next_token()
    print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
```

No multimodal inference script is shipped yet: the VL package has not been
produced once (see the blocker above), so any image-prompt driver would be
written against an unverified `genai_config.json` / processor contract. It will
be added after the first successful export, together with the measured results.

## Status

- `rtn_fp16`: RTN 4-bit MoE quantization + Mobius ONNX export was **validated
  end to end** on a single A100-80GB GPU (~4 minutes wall time for quantization
  + export), and `onnxruntime_genai.Model(...)` load + generation was validated.
  That run predates Olive#2630; re-validation with the pinned mixed-precision
  dependencies is pending.
- `kquant_fp16`: **config only.** It is the same workflow as `rtn_fp16` with
  the `KQuant` pass substituted, and it parses/validates against Olive's
  `RunConfig` schema, but no full quantize + export + generate run has been
  recorded yet. Expect the k-quant search to be substantially slower and use
  more transient device memory than RTN on the fused expert tensors.
- `vl_kquant_fp16`: **config only, and blocked** on
  [mobius#515](https://github.com/onnxruntime/mobius/pull/515). The Olive-side
  half (VL class loading, composite config resolution, decoder-only
  quantization target selection) is covered by
  [Olive#2630](https://github.com/microsoft/Olive/pull/2630) and was verified
  there against this checkpoint's real config plus a shape-faithful synthetic
  VL model; the end-to-end export on the real 35B checkpoint has **not** been
  run. On the first run, verify that Olive copied the pinned processor metadata
  into the quantized checkpoint and that mobius's emitted
  `processor_config.json` matches it. Peak host/GPU memory for loading the VL
  class (decoder + vision tower) is also unmeasured.
- GPTQ MoE quantization on this model was evaluated but deferred: with 40
  layers x 256 experts (10240 total per-expert Cholesky solves), GPTQ is
  estimated to take multiple hours on a single GPU (Olive's GPTQ pass has no
  multi-GPU parallelism), so RTN was prioritized first to validate the
  export/runtime pipeline end to end.

## References

- Mobius: <https://github.com/onnxruntime/mobius>
- mobius#505 (QMoE scale precision): <https://github.com/onnxruntime/mobius/pull/505>
- mobius#511 (VL decoder QMoE support): <https://github.com/onnxruntime/mobius/pull/511>
- mobius#515 (Olive mixed-precision VL/MoE export — **required** for `vl_kquant_fp16`): <https://github.com/onnxruntime/mobius/pull/515>
- Olive#2630 (Qwen3.5/3.6-MoE VL checkpoints in PyTorch-side quantization): <https://github.com/microsoft/Olive/pull/2630>
- Olive#2610 (exclude MoE routers from quantization): <https://github.com/microsoft/Olive/pull/2610>
- Olive `Rtn` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/pytorch/rtn.py>
- Olive `KQuant` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/pytorch/kquant.py>
- Olive `MobiusBuilder` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py>
