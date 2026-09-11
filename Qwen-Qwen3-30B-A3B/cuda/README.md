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
512 non-overlapping blocks of 2048 tokens. The dataset is pinned to revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. GPTQ quality is
calibration-dependent; use the same calibration policy when comparing runs.
Experts with insufficient routed calibration tokens automatically fall back
to RTN, so retain the fallback count from the GPTQ log with benchmark results.

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

Use ONNX Runtime GenAI's upstream
[`model-qa.py`](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py)
sample. Run it from the `Qwen-Qwen3-30B-A3B/` directory, replacing
`/path/to/onnxruntime-genai` with a checkout of that repository:

```bash
# Single prompt; /no_think suppresses Qwen3's reasoning trace
python /path/to/onnxruntime-genai/examples/python/model-qa.py \
  --model_path cuda/kquant_fp16/models \
  --execution_provider cuda \
  --user_prompt "What is 17 * 23? /no_think" \
  --non_interactive \
  --timings

# Interactive, stateless question/answer loop
python /path/to/onnxruntime-genai/examples/python/model-qa.py \
  -m cuda/kquant_fp16/models \
  -e cuda \
  --timings
```

The sample applies the package's chat template and removes each user message
after generation, so interactive prompts are independent and retain only the
configured system prompt. `--timings` reports time to first token plus prompt
and new-token throughput; `--verbose` shows model and search setup.
`--max_length` can cap the total prompt-plus-generation length. Reasoning
traces can be long, so include Qwen3's `/no_think` switch directly in a prompt
when a short answer is enough. Change `--model_path` to any other output path
in the table above to test that variant.

This is a standalone decoder-only text package, so ORT GenAI uses its text-only
runtime path; no vision encoder, image processor, or multimodal pipeline is
involved.

## Evaluation

`eval/mmlu_cuda.json` declaratively scores the standalone ORT GenAI package
with Olive's `LMEvaluator` and
[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). It
defaults to the KQuant output and preserves the benchmark settings used for
the results below: MMLU, 200 samples per subtask, task-default few-shot
behavior (`num_fewshot: null`), batch size 1, maximum length 4096, the CUDA
execution provider, the `ortgenai` model class, and no chat-template wrapping.

Run from the `Qwen-Qwen3-30B-A3B/` directory. Olive resolves the config's
relative model path from the current working directory:

```bash
# KQuant (the config default)
olive run --config eval/mmlu_cuda.json

# Run exactly the same evaluator against all four exported variants
for model_path in \
  cuda/fp16/models \
  cuda/kquant_fp16/models \
  cuda/rtn_fp16/models \
  cuda/gptq_fp16/models
do
  olive run --config eval/mmlu_cuda.json \
    --model_name_or_path "$model_path"
done
```

For an ad-hoc run without the checked-in config, Olive's benchmark command has
equivalent defaults for task-selected few-shot behavior and chat wrapping:

```bash
for model_path in \
  cuda/fp16/models \
  cuda/kquant_fp16/models \
  cuda/rtn_fp16/models \
  cuda/gptq_fp16/models
do
  olive benchmark \
    --model_name_or_path "$model_path" \
    --tasks mmlu \
    --backend ortgenai \
    --device gpu \
    --batch_size 1 \
    --max_length 4096 \
    --limit 200
done
```

The FP16 result is the unquantized quality baseline; keep all evaluator options
identical when comparing variants. This is a text `lm-eval` path, not the
`lmms-eval` path used by vision-language models.

> lm-eval downloads its task datasets from the Hugging Face Hub on first use, so
> the machine needs network access (and `huggingface-cli login` for gated
> datasets). Set `HF_HOME` or `HF_DATASETS_CACHE` to reuse an existing cache.

## Benchmark results

All four variants were exported and evaluated end to end on a single NVIDIA
A100 80 GB GPU (CUDA execution provider) with ONNX Runtime 1.30.0 and ONNX
Runtime GenAI 0.16.0-dev. MMLU was run identically across all four model
directories using the methodology now encoded in `eval/mmlu_cuda.json`: limit
200 per subtask, task-default few-shot setting, batch size 1, maximum length
4096, the `ortgenai` model class on CUDA, and no chat-template wrapping:

| Variant | MMLU acc | acc_stderr | Δ vs FP16 | Export time |
|---|---|---|---|---|
| FP16 (baseline, unquantized) | 0.8077 | 0.0039 | — | ~16 min |
| GPTQ | 0.7982 | 0.0040 | -0.95 pt | ~2 h |
| KQuant | 0.7963 | 0.0040 | -1.14 pt | few min |
| RTN | 0.7918 | 0.0040 | -1.59 pt | ~6 min |

(9,183 effective samples out of 57 MMLU subtasks; subtasks with fewer than 200
test examples were run to completion rather than padded.)

The GPTQ row above is the original 128-sample baseline. A pinned replication
produced identical external tensor data and the same rounded accuracy and
fallback counts. Accuracy degrades in the expected direction (FP16 > GPTQ >
KQuant > RTN), and
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

The checked-in GPTQ recipe now uses 512 samples following a bounded
128→256→512 scaling study with the dataset revision above and otherwise
identical quantization, preprocessing, and MMLU settings. Every generation
smoke test produced `391`. MMLU changes are much smaller than the combined
standard errors, while gate/up fallback coverage improves materially:

| Calibration blocks | Input tokens | Gate/up fallback | Down fallback | MMLU accuracy | Export pass duration |
|---:|---:|---:|---:|---:|---:|
| 128 | 262,144 | 2,438/6,144 (39.68%) | 1,821/6,144 (29.64%) | 79.82% ± 0.40% | 1 h 54 min 43 s |
| 256 | 524,288 | 1,878/6,144 (30.57%) | 1,640/6,144 (26.69%) | 79.60% ± 0.40% | 2 h 11 min 6 s |
| 512 | 1,048,576 | 1,657/6,144 (26.97%) | 1,657/6,144 (26.97%) | 79.61% ± 0.40% | 2 h 57 min 7 s |

The 512 result is selected because it passes the smoke and MMLU quality gates
and reduces gate/up fallback by another 3.60 percentage points from 256. The
paired per-example comparison over the same 9,183 MMLU examples found no
statistically significant accuracy difference:

| Comparison | Accuracy delta | Paired 95% confidence interval |
|---|---:|---:|
| 128 → 256 | -0.22 pp | [-0.67, +0.23] pp |
| 256 → 512 | +0.01 pp | [-0.45, +0.47] pp |

The selection therefore maximizes measured calibration coverage under the
bounded study; it does not establish 512 samples as a unique MMLU optimum. The
current ORT GenAI lm-eval adapter does not implement rolling loglikelihood, so
comparable WikiText test perplexity was unavailable and no proxy was used.

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
