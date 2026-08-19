# Qwen3.6-35B-A3B VL KQuant + Mobius (CUDA)

This recipe exports
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
as a three-component ONNX Runtime GenAI vision-language package for CUDA.

The workflow uses Olive's MoE-aware PyTorch-side `KQuant` pass for supported
decoder weights, followed by `MobiusBuilder` to export the decoder, vision
encoder, embedding model, runtime configuration, tokenizer, and image
processor in one operation.

## Quantization scope

The relevant `KQuant` options are explicit in `vl_kquant_fp16/config.json`:

```json
{
    "bits": 4,
    "group_size": 128,
    "sym": true,
    "moe": true,
    "quantize_vision": false,
    "embeds": false,
    "lm_head": false
}
```

`quantize_vision` is the Olive option controlling whether the vision tower
participates in this PyTorch-side quantization pass. There is no `vision_only`
option. Setting it to `false` leaves the vision tower floating point.
`embeds: false` and `lm_head: false` likewise leave the input embedding and
language-model head floating point.

The resulting mixed-precision selection is:

| Module group | Representation |
|---|---|
| Decoder self-attention linears | int4 KQuant |
| Shared-expert MLP linears | int4 KQuant |
| Fused MoE expert tensors | int4 KQuant, independently per expert |
| Vision tower | fp16 export (`quantize_vision: false`) |
| Input embedding and LM head | fp16 export (`embeds: false`, `lm_head: false`) |
| MoE routers and `shared_expert_gate` | fp16 export; always excluded by Olive |
| GatedDeltaNet `linear_attn` | fp16 export; recurrent/SSM blocks are excluded by Olive |

The `fp16` descriptions refer to the Mobius export precision. The original
checkpoint values remain at their loaded floating-point precision until
export.

## Why one workflow is sufficient

The input uses `task: image-text-to-text`, so Transformers loads
`Qwen3_5MoeForConditionalGeneration` with both the decoder and vision tower.
Olive applies the selection above to the full model and saves one mixed
checkpoint.

Mobius then exports the complete ORT GenAI package:

- `decoder`: hybrid full/GatedDeltaNet attention and quantized MoE FFNs
- `vision_encoder`: floating-point Qwen vision transformer
- `embedding`: floating-point token embedding and image-feature fusion

No partial export, per-component Olive workflow, or manual
`genai_config.json` assembly is required.

## Required revisions

`requirements.txt` pins unreleased revisions containing:

- Olive MoE-aware KQuant and Qwen3.5/3.6 VL selection support
- [mobius#515](https://github.com/onnxruntime/mobius/pull/515), which builds
  Olive-format Qwen3.5/3.6 `linear_attn` and `shared_expert_gate` as floating
  point while retaining quantized attention/MLP/QMoE modules
- [mobius#519](https://github.com/onnxruntime/mobius/pull/519), which treats
  both the outer `qwen3_5_moe` config and unwrapped `qwen3_5_moe_text` config
  as Qwen VL processor targets, ensuring `PatchImage` produces the rank-2
  packed patches expected by the vision encoder
- `onnx-ir>=1.0.0`, required by Mobius's sharded external-data save API

Replace the Git revisions with compatible releases after those changes ship.

## Why CUDA uses fp16

The target system and CUDA execution provider are explicit because
`MobiusBuilder` derives graph/runtime choices from the Olive accelerator
specification.

The export uses `fp16`, not `bf16`. ONNX Runtime's CUDA
`CausalConvWithState` kernel used by Qwen3.6 GatedDeltaNet layers currently
supports float and float16, but not bfloat16. Mobius also emits QMoE scales in
the requested export precision so QMoE remains assigned to CUDA.

## Setup and export

```bash
pip install -r cuda/requirements.txt

olive run --config cuda/vl_kquant_fp16/config.json
```

Run the command from the `Qwen-Qwen3.6-35B-A3B/` directory because
`output_dir` is relative to that directory.

## Output

```text
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
├── genai_config.json
├── processor_config.json
├── tokenizer.json
├── tokenizer_config.json
└── chat_template.jinja
```

## Inference

Image and text:

```bash
python cuda/inference_vl.py \
  --image /path/to/image.png \
  --prompt "Describe this image."
```

The same package also supports text-only prompts, interactive input, and a
directory benchmark reporting time-to-first-token and decode throughput:

```bash
python cuda/inference_vl.py --prompt "What is the capital of France?"
python cuda/inference_vl.py --interactive
python cuda/inference_vl.py --benchmark /path/to/images
```

## Evaluation

`eval_vl.py` measures diagram-question accuracy and average latency on a
deterministic prefix of the AI2D test set:

```bash
python cuda/eval_vl.py --num-samples 100
```

AI2D is downloaded from Hugging Face on first use. The script evaluates the
exported ORT GenAI package only; it does not co-load the roughly 70 GB fp16
PyTorch checkpoint on the same GPU.

## Validation

The recipe was validated end to end on one NVIDIA A100-SXM4-80GB:

- KQuant selected 240 decoder parameters and completed in about 194 seconds
- Mobius export completed in about 158 seconds
- the three-component package was approximately 22 GB
- `onnxruntime_genai.Model` loaded the package on CUDA in about 59 seconds
- a synthetic 256x256 four-color image produced 86 input tokens and successful
  multimodal generation; the model correctly identified a square divided into
  four equal quadrants

The expanded inference benchmark modes and full AI2D evaluation have not yet
been run against the large artifact because it was removed after validation.

## References

- [Olive KQuant](https://github.com/microsoft/Olive/tree/main/olive/passes/pytorch/kquant.py)
- [Olive MobiusBuilder](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
- [Olive#2630: Qwen3.5/3.6 MoE VL quantization selection](https://github.com/microsoft/Olive/pull/2630)
- [Mobius](https://github.com/onnxruntime/mobius)
- [mobius#505: QMoE scale precision](https://github.com/onnxruntime/mobius/pull/505)
- [mobius#511: VL decoder QMoE support](https://github.com/onnxruntime/mobius/pull/511)
- [mobius#515: Olive mixed-precision export](https://github.com/onnxruntime/mobius/pull/515)
- [mobius#519: Qwen3.6 packed VL processor](https://github.com/onnxruntime/mobius/pull/519)
