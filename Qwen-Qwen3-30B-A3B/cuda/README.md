# Qwen3-30B-A3B weight-only quantization + Mobius (CUDA)

This recipe exports
[`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B)
as an ONNX Runtime GenAI text-generation package for CUDA.

The workflows export an unquantized fp16 baseline or apply Olive's PyTorch-side
`KQuant`, `Rtn`, or `Gptq` pass to the dense decoder linears and fused MoE
expert weights. `MobiusBuilder` then produces a decoder-only model and its ORT
GenAI configuration.

## Quantization

All three quantized recipes use symmetric 4-bit weights with group size 128:

```json
{
    "bits": 4,
    "group_size": 128,
    "sym": true,
    "moe": true,
    "embeds": false,
    "lm_head": false
}
```

The 128 routed experts in every MoE layer are quantized independently. Routers,
token embeddings, normalization layers, and the language-model head remain
floating point. Mobius exports floating-point tensors and activations at fp16
precision.

`KQuant` and `Rtn` do not require calibration data. `Gptq` loads the WikiText-2
`train` split, joins its leading rows into a token stream, and uses the first
128 non-overlapping blocks of 2048 tokens. GPTQ quality is
calibration-dependent; use the same calibration policy when comparing runs.
Experts with insufficient routed calibration tokens automatically fall back to
RTN, so retain the fallback count from the GPTQ log with benchmark results.

## Setup and export

Run from the `Qwen-Qwen3-30B-A3B/` directory:

```bash
pip install -r cuda/requirements.txt

# Choose one workflow
olive run --config cuda/fp16/config.json
olive run --config cuda/kquant_fp16/config.json
olive run --config cuda/rtn_fp16/config.json
olive run --config cuda/gptq_fp16/config.json
```

Each workflow writes to its own directory:

| Variant | Configuration | Output |
|---|---|---|
| FP16 baseline (~61 GB) | `cuda/fp16/config.json` | `cuda/fp16/models/` |
| KQuant | `cuda/kquant_fp16/config.json` | `cuda/kquant_fp16/models/` |
| RTN | `cuda/rtn_fp16/config.json` | `cuda/rtn_fp16/models/` |
| GPTQ | `cuda/gptq_fp16/config.json` | `cuda/gptq_fp16/models/` |

## Inference

`inference.py` in the recipe root loads the exported package with ONNX Runtime
GenAI and streams a greedy response. Run it from the `Qwen-Qwen3-30B-A3B/`
directory:

```bash
# Single prompt (reasoning enabled, the Qwen3 default)
python inference.py --prompt "What is the capital of France?"

# Skip the <think> reasoning trace with Qwen3's /no_think switch
python inference.py --prompt "What is 17 * 23?" --no-think

# Stateless interactive loop (each turn is an independent single-turn chat)
python inference.py --interactive
```

The script defaults to `--model-path cuda/kquant_fp16/models`, generates
`--max-new-tokens 1024` tokens on top of the prompt, and reports time to first
token, decode throughput, and total generation time. Reasoning traces can be
long, so raise `--max-new-tokens` or pass `--no-think` when a short answer is
enough. `--system-prompt` sets a system turn and `--verbose` adds the input
token count and the resolved search length. Pass `--model-path cuda/fp16/models`,
`--model-path cuda/rtn_fp16/models`, or `--model-path
cuda/gptq_fp16/models` to test another variant.

The generic ORT GenAI text-generation example works as well:

```bash
python /path/to/onnxruntime-genai/examples/python/model-generate.py \
  -m cuda/kquant_fp16/models \
  -e cuda \
  -pr "What is the capital of France?" \
  --non_interactive
```

This is a decoder-only package, so ORT GenAI uses its text-only runtime path;
no vision encoder, image processor, or multimodal pipeline is involved.

## Evaluation

`eval.py` in the recipe root scores the exported package with
[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) through
Olive's `ortgenai` model class (`olive.evaluator.lmeval_ort`) on the CUDA
execution provider. Run it from the `Qwen-Qwen3-30B-A3B/` directory:

```bash
# MMLU, 100 samples per subtask (defaults)
python eval.py

# A single, quicker task
python eval.py --task arc_challenge --limit 200

# Full task, no sample cap
python eval.py --task mmlu --limit 0
```

`--limit` caps the samples per task (task groups such as `mmlu` apply it to
every subtask), `--max-length` sets the evaluation sequence length, and
`--num-fewshot` overrides the shot count baked into the task. Every metric
lm-eval reports for the task is printed. Use `--model-path` to evaluate the
corresponding FP16, KQuant, RTN, or GPTQ output. The FP16 result is the
unquantized quality baseline; use identical task, limit, and few-shot settings
for all four variants.

> lm-eval downloads its task datasets from the Hugging Face Hub on first use, so
> the machine needs network access (and `huggingface-cli login` for gated
> datasets). Set `HF_HOME` or `HF_DATASETS_CACHE` to reuse an existing cache. No
> accuracy numbers have been measured for this recipe yet.

## KQuant validation status

The complete KQuant workflow was validated on an NVIDIA A100 80 GB GPU with ONNX
Runtime 1.30.0 and ONNX Runtime GenAI 0.16.0-dev:

- KQuant produced a 16.7 GB quantized checkpoint.
- Mobius exported a 16.7 GB `model.onnx.data` file and the ORT GenAI package.
- ORT GenAI loaded the package in approximately 53-58 seconds.
- CUDA greedy generation produced coherent text, valid Python code, and the
  correct answer `391` for `17 * 23`.
- Qwen3 reasoning token metadata resolved correctly (`bor=151667`,
  `eor=151668`).

Observed generation throughput ranged from 61 to 182 output tokens per second
for the short smoke-test prompts after model load. This is a functional
validation result, not a controlled performance benchmark. The FP16, RTN, and
GPTQ variants are provided for controlled comparison but have not yet been run
end to end on this model.

## References

- [Olive KQuant](https://github.com/microsoft/Olive/blob/main/olive/passes/pytorch/kquant.py)
- [Olive RTN](https://github.com/microsoft/Olive/blob/main/olive/passes/pytorch/rtn.py)
- [Olive GPTQ](https://github.com/microsoft/Olive/blob/main/olive/passes/pytorch/gptq.py)
- [Mobius](https://github.com/onnxruntime/mobius)
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)
