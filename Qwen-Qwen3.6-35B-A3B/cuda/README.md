# Qwen3.6-35B-A3B text-only KQuant + Mobius (CUDA)

This recipe exports the text model from
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
as a standalone ONNX Runtime GenAI package for the CUDA execution provider.
The vision encoder is intentionally not exported.

## Pipeline and quantization scope

Olive loads the checkpoint with `task: text-generation`, selecting the
standalone `qwen3_5_moe_text` causal language model. The pipeline then:

1. Applies the MoE-aware `KQuant` pass to supported decoder weights using
   symmetric 4-bit quantization with group size 128.
2. Uses `MobiusBuilder` to export the quantized text model and its ONNX Runtime
   GenAI configuration and tokenizer assets at fp16 precision.

The token embeddings and language-model head are excluded from KQuant by
`embeds: false` and `lm_head: false`. MoE routers, `shared_expert_gate`, and
the GatedDeltaNet `linear_attn` recurrent/SSM blocks also remain floating
point rather than being KQuant targets. Because this is a text-generation
workflow, it neither quantizes nor exports a vision tower, image embedding
model, image processor, or other multimodal component.

## Why CUDA uses fp16

The target `LocalSystem` explicitly selects `CUDAExecutionProvider`, allowing
Mobius to build the graph and runtime configuration for CUDA. The Mobius
export precision is fp16 because that is the precision validated for the
complete CUDA graph. The exported artifact uses standard `Conv` nodes after
Mobius inlining; bf16 was not validated for the complete graph.

## Setup and export

From the `Qwen-Qwen3.6-35B-A3B/` directory:

```bash
pip install -r cuda/requirements.txt
olive run --config cuda/kquant_fp16/config.json
```

The pinned model revision makes the source checkpoint reproducible. The
dependency revisions provide Olive's MoE-aware KQuant and MobiusBuilder
support. Mobius PR #525 provides packed Olive QMoE sidecar handling, while
Mobius PR #631 is required for correct generic-decoder versus multimodal
runtime routing. PR #631 is still open and unmerged, so its revision remains
pinned by `requirements.txt`.

## Runtime setup

`requirements.txt` covers export dependencies only. The documented inference
result requires ONNX Runtime GenAI built from source at commit
`a8e0fdf81b061e67c1c3f9485bfdc06735ccd473` together with the
`onnxruntime-gpu` 1.30.0.dev20260823001 nightly built from ONNX Runtime commit
`4d308dacbb`. Follow the upstream
[ONNX Runtime GenAI build instructions](https://github.com/microsoft/onnxruntime-genai/blob/main/BUILD.md)
for the source build.

Stable ONNX Runtime GenAI 0.15.2 was not validated for this graph. The
generated `runtime_compatibility.json` values (minimum 0.14.0 and tested
0.15.2) are generic decoder ABI metadata and do not represent
model-specific runtime validation.

## Output

The validated export is a root-level standalone text package:

```text
cuda/kquant_fp16/models/
├── footprint.json
├── output_footprint.json
├── run_history.txt
├── model_config.json
├── model.onnx
├── model.onnx.data
├── genai_config.json
├── runtime_compatibility.json
├── chat_template.jinja
├── tokenizer.json
└── tokenizer_config.json
```

There is no `vision_encoder`, embedding model, or image processor. The package
is 20 GB; `model.onnx.data` is 20,948,779,008 bytes.

## Text inference

After export, validate and test the package with the standard ONNX Runtime
GenAI
[`model-chat.py`](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py)
or
[`model-qa.py`](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py)
examples. For example, from the recipe directory:

```bash
export ORT_GENAI_ROOT=/path/to/onnxruntime-genai
python "$ORT_GENAI_ROOT/examples/python/model-qa.py" \
  -m cuda/kquant_fp16/models/ \
  --user_prompt "Calculate 17 * 23 and state the final answer." \
  --max_length 512 \
  --non_interactive
```

The default chat template enables reasoning mode. Allow enough `max_length`
for reasoning prompts so generation can reach the final answer.

## Text evaluation

Use Olive's standard lm-eval integration for text benchmarks instead of a
model-specific evaluation script. Install the optional benchmark dependency,
then evaluate the exported ORT GenAI package:

```bash
pip install lm-eval
olive benchmark \
  --model_name_or_path cuda/kquant_fp16/models/ \
  --tasks mmlu \
  --device gpu \
  --backend ortgenai \
  --batch_size 1 \
  --max_length 4096 \
  --limit 100 \
  --output_path benchmark/mmlu
```

This exercises only the standalone decoder with text lm-eval tasks. It is not
a VLM benchmark and does not require a multimodal processor. No accuracy result
from this optional evaluation command is claimed in the validation below.

## Validation

The complete recipe was validated on 2026-08-25 with one NVIDIA
A100-SXM4-80GB and the CUDA execution provider:

- Olive loaded `Qwen/Qwen3.6-35B-A3B` at the pinned revision beginning
  `995ad96e` as `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text` with Transformers
  5.15.1 and the `text-generation` task.
- KQuant used symmetric int4 quantization with group size 128 and MoE support
  enabled, while excluding embeddings and the language-model head. It
  quantized 240 parameter tensors in 500.967169 seconds.
- The fp16 MobiusBuilder export completed in 143.995098 seconds.
- `genai_config.json` identifies a `decoder` model with root-level
  `model.onnx`, enables CUDA graphs with `enable_cuda_graph=1`, and sets a
  262144-token context length.
- The generated `runtime_compatibility.json` records generic-decoder minimum
  ONNX Runtime GenAI 0.14.0 and tested version 0.15.2. Model-specific runtime
  validation instead used source-built ONNX Runtime GenAI 0.16.0.dev0 at commit
  `a8e0fdf81b061e67c1c3f9485bfdc06735ccd473` and `onnxruntime-gpu`
  1.30.0.dev20260823001 from commit `4d308dacbb`.
- Exactly two reasoning-mode smoke runs used `model-qa.py` with greedy
  decoding (`top_k=1`). Both produced coherent reasoning that derived `391`
  for `Calculate 17 * 23 and state the final answer.`, but the capped runs
  were not evidence of completed final-channel answers.
- In a separate no-think check, the Hugging Face chat template was rendered
  with `enable_thinking=false`; generation produced the exact output `391`.
  This check was repeated successfully in the validated `qwen35-4b`
  environment.

The two reasoning-mode runs produced the following smoke measurements on that
one A100. These are not a broad benchmark:

| Max total tokens (prompt + generated cap) | Time to first token | Decode throughput |
|---:|---:|---:|
| 256 | 4.25 s | 77.66 tokens/s |
| 512 | 3.36 s | 93.16 tokens/s |

## References

- [Olive KQuant](https://github.com/microsoft/Olive/tree/main/olive/passes/pytorch/kquant.py)
- [Olive MobiusBuilder](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
- [Mobius](https://github.com/onnxruntime/mobius)
- [Mobius PR #525: packed Olive QMoE sidecars](https://github.com/onnxruntime/mobius/pull/525)
- [Mobius PR #631: standalone text and multimodal runtime type routing](https://github.com/onnxruntime/mobius/pull/631)
