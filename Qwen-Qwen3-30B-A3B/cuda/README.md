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

> `cuda/requirements.txt` leaves `onnxruntime-genai-cuda` unpinned. As of this
> writing, neither the ONNX Runtime 1.30.0 nor the ONNX Runtime GenAI
> 0.16.0-dev build used for the validation below (see "Benchmark results" and
> "KQuant validation status") is published on PyPI or the public
> [ORT-Nightly feed](https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/)
> yet — both were built from source. `pip install -r cuda/requirements.txt`
> alone will resolve to whatever `onnxruntime-genai-cuda` release is current
> on PyPI, which may be older and behave differently. To reproduce the
> validated setup exactly, build `onnxruntime` and `onnxruntime-genai` from
> source for CUDA (see the
> [ONNX Runtime GenAI build docs](https://github.com/microsoft/onnxruntime-genai/blob/main/README.md#build-from-source))
> and install the resulting wheels before running `olive run`. Once a public
> release incorporates the required functionality, pin
> `onnxruntime-genai-cuda` (and `onnxruntime-gpu`, if needed) here instead.

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
> datasets). Set `HF_HOME` or `HF_DATASETS_CACHE` to reuse an existing cache.

## Benchmark results

All four variants were exported and evaluated end to end on a single NVIDIA
A100 80 GB GPU (CUDA execution provider) with ONNX Runtime 1.30.0 and ONNX
Runtime GenAI 0.16.0-dev. `eval.py --task mmlu --limit 200` was run identically
across all four model directories (same task, sample cap, and few-shot
setting):

| Variant | MMLU acc | acc_stderr | Δ vs FP16 | Export time |
|---|---|---|---|---|
| FP16 (baseline, unquantized) | 0.8077 | 0.0039 | — | ~16 min |
| GPTQ | 0.7982 | 0.0040 | -0.95 pt | ~2 h |
| KQuant | 0.7963 | 0.0040 | -1.14 pt | few min |
| RTN | 0.7918 | 0.0040 | -1.59 pt | ~6 min |

(9,183 effective samples out of 57 MMLU subtasks; subtasks with fewer than 200
test examples were run to completion rather than padded.)

Accuracy degrades in the expected direction (FP16 > GPTQ > KQuant > RTN), and
all three quantized variants stay within ~1.6 points of the unquantized
baseline at 4-bit weights. GPTQ improves over plain RTN by 0.64 points despite
a substantial calibration-coverage shortfall: across the 48 MoE layers (6,144
routed experts total), **2,344 experts (38.2%) were "starved"** of calibration
tokens and **94 (1.5%) were entirely unseen**, so roughly 40% of all experts in
the "GPTQ" checkpoint were actually quantized with the RTN fallback, not GPTQ.
This is with the default calibration policy (WikiText-2, 128 samples x 2048
tokens) — a larger/more diverse calibration set that routes tokens to more
experts should close more of this coverage gap and is expected to further
improve GPTQ's result. GPTQ's ~2 hour calibration/quantization time (vs. a few
minutes for KQuant/RTN) is the dominant cost of producing all four variants;
eval itself takes ~5-6 minutes per quantized variant and ~75 minutes for the
uncompressed FP16 baseline (memory-bandwidth bound, not compute bound).

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
validation result, not a controlled performance benchmark. See "Benchmark
results" above for the accuracy comparison across all four variants, which
have now all been run end to end on this model.

## References

- [Olive KQuant](https://github.com/microsoft/Olive/blob/main/olive/passes/pytorch/kquant.py)
- [Olive RTN](https://github.com/microsoft/Olive/blob/main/olive/passes/pytorch/rtn.py)
- [Olive GPTQ](https://github.com/microsoft/Olive/blob/main/olive/passes/pytorch/gptq.py)
- [Mobius](https://github.com/onnxruntime/mobius)
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)
