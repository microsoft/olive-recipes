# Qwen3.6-35B-A3B (Olive RTN MoE + Mobius + ONNX Runtime GenAI, CUDA)

This folder contains an Olive recipe for exporting the text-only decoder of
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — a hybrid
MoE model (40 layers, 256 experts, 8 experts/token, GatedDeltaNet linear
attention interleaved with full attention) — to ONNX for ONNX Runtime GenAI on
the CUDA execution provider.

Unlike the `NvTensorRtRtx` recipe in this folder (which uses ORT GenAI's
`ModelBuilder` + `MatMulNBitsToQDQ`), this recipe uses Olive's newer
MoE-aware quantization + [`MobiusBuilder`](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
export pipeline (via [mobius](https://github.com/onnxruntime/mobius)),
which natively understands per-expert MoE weight layouts and hybrid
linear/full-attention graphs.

## Pipeline

1. **`Rtn` pass** (`olive/passes/pytorch/rtn.py`): 4-bit symmetric RTN
   weight-only quantization, `group_size=128`, with `moe=true` so each of the
   256 experts per layer is quantized independently (rather than merging all
   experts into a single blockwise-quantized tensor).
2. **`MobiusBuilder` pass**: exports the quantized PyTorch checkpoint straight
   to a full ORT GenAI package (`model.onnx` + `model.onnx.data` +
   `genai_config.json` + tokenizer files) — no intermediate ONNX conversion
   pass is needed.

## Why `precision: fp16` (not `bf16`) and why `target`/`accelerators` matter

Two footguns we hit while building this recipe, both worth calling out
explicitly since they are easy to reproduce with a naive config:

- **EP must be set explicitly via `systems`/`target`.** `MobiusBuilder` reads
  the execution provider from the Olive accelerator spec
  (`self.accelerator_spec.execution_provider`), not from a pass parameter. If
  `systems`/`target` are omitted, Olive defaults to `cpu-cpu`, and the CPU EP
  only fuses `GroupQueryAttention` for `fp32` — `fp16`/`bf16` on CPU silently
  fall back to the standard (non-GQA) `Attention` op, which is incompatible
  with the `past_present_share_buffer=True` mode required by this model's
  `LinearAttention` (recurrent-state) layers. Mobius raises a clear
  `ValueError` in that case rather than emitting a broken `genai_config.json`
  — the fix is to point the workflow at the CUDA EP (see `config.json`) and
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
- **Disable `enable_cuda_graph` for now.** The genai_config the CUDA EP
  capabilities produce sets `"enable_cuda_graph": "1"` by default,
  but not all decoder graph nodes are currently assigned to the CUDA EP
  (some fall back to CPU), and ORT GenAI's CUDA graph capture requires the
  *entire* graph to run on one EP. Until that gap is closed, remove
  `enable_cuda_graph` from the generated `genai_config.json`'s
  `provider_options` before loading the model with `onnxruntime_genai`, or
  model load will fail with: `This session cannot use the graph capture
  feature as requested by the user as all compute graph nodes have not been
  partitioned to the CUDAExecutionProvider`.

## Setup

```bash
pip install olive-ai[gpu] mobius-onnx onnxruntime-genai-cuda
```

## Run

```bash
olive run --config cuda/rtn_fp16/config.json
```

This produces the full ORT GenAI package under `cuda/rtn_fp16/models/`:

```
cuda/rtn_fp16/models/
├── model.onnx          # ONNX graph
├── model.onnx.data     # External weight data (~20GB for this model)
├── genai_config.json   # ORT GenAI runtime configuration
├── tokenizer.json
├── tokenizer_config.json
└── chat_template.jinja
```

## Verify with `onnxruntime_genai`

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

## Status

- RTN 4-bit MoE quantization + Mobius ONNX export: **validated end to end** on
  a single A100-80GB GPU (~4 minutes wall time for quantization + export).
- `onnxruntime_genai.Model(...)` load + generation: validated after applying
  the `enable_cuda_graph` workaround above.
- GPTQ MoE quantization on this model was evaluated but deferred: with 40
  layers x 256 experts (10240 total per-expert Cholesky solves), GPTQ is
  estimated to take multiple hours on a single GPU (Olive's GPTQ pass has no
  multi-GPU parallelism), so RTN was prioritized first to validate the
  export/runtime pipeline end to end.

## References

- Mobius: <https://github.com/onnxruntime/mobius>
- Olive `Rtn` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/pytorch/rtn.py>
- Olive `MobiusBuilder` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py>
