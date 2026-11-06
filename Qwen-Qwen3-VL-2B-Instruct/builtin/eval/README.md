# Qwen3-VL-2B-Instruct Built-in Olive Evaluation

Evaluate the complete Mobius packages with Olive's built-in vision evaluator
and `LMEvaluator`. No recipe-specific evaluation or scoring callbacks are used.

## Benchmarks

- **AI2D** — Science diagram multiple-choice QA (`lmms-lab/ai2d`, test split)
- **MMLU** - Five-shot text multiple-choice accuracy, all 57 subjects.

## Prerequisites

Export the complete package first (see [export instructions](../README.md)).
Use the matching ORT GenAI runtime required by runtime last-token indices.
Install the recipe requirements, including `lm-eval` and `evaluate`.
Run the commands below from this `eval` directory.

## Usage

```bash
# AI2D: first 100 test samples, same protocol on both devices
olive run --config ai2d_cpu.json
olive run --config ai2d_cuda.json

# MMLU: limit 100 PER SUBJECT, not 100 total
olive run --config mmlu_cpu.json
olive run --config mmlu_cuda.json
```

## Configs

| Config | Model Path | Device |
|--------|-----------|--------|
| [ai2d_cpu.json](ai2d_cpu.json) | `../cpu_and_mobile/models` | CPU |
| [ai2d_cuda.json](ai2d_cuda.json) | `../cuda/models` | GPU (CUDA) |
| [mmlu_cpu.json](mmlu_cpu.json) | `../cpu_and_mobile/models` | CPU |
| [mmlu_cuda.json](mmlu_cuda.json) | `../cuda/models` | GPU (CUDA) |

AI2D uses `vision_vqa_pre_process`, the original images, numbered choices,
a single-digit system prompt, greedy decoding, and a 4,096-token total limit.
`exact_match` compares Olive's extracted option number with the target.
This differs from the removed custom evaluator's eight-new-token protocol.

MMLU uses the incremental `ortgenai` backend at batch size 1, five-shot chat
prompts, and a 4,096-token context. Last-token-only logits are supported.
CUDA keeps shared past/present buffers enabled because Qwen-VL decoder
graph capture requires them.
For a 57-question smoke run, use a private copy with `limit: 1`. That is not
a full MMLU score.

For baseline comparisons, copy a workflow and change only `input_model` to the
baseline package (`text.onnx` for the older flat layout). Keep the dataset,
sample selection, prompts, lengths, and runtime identical. Historical README
scores from the custom evaluator are not directly comparable.
