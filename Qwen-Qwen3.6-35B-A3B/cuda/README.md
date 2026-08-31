# Qwen3.6-35B-A3B standalone text four-way comparison (CUDA)

These recipes export the text model from the VLM-capable
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
checkpoint as standalone ONNX Runtime GenAI packages for the CUDA execution
provider. They provide an unquantized FP16 baseline and KQuant, RTN, and GPTQ
weight-only quantized variants. Vision is intentionally omitted from every
variant so that all four workflows compare the same causal language model.

## Recipe and package layout

The provider-level setup, metadata, and documentation remain under `cuda/`,
while the four export configurations and shared evaluator are organized under
the `cuda/text/` path namespace:

```text
cuda/
├── README.md
├── info.yml
├── requirements.txt
└── text/
    ├── fp16/config.json
    ├── kquant_fp16/config.json
    ├── rtn_fp16/config.json
    ├── gptq_fp16/config.json
    └── eval/mmlu_cuda.json
```

`text/` is only a recipe path namespace; it is not a Mobius component. All four
input models set `task: text-generation`, and all four MobiusBuilder passes
explicitly set `text_only: true`. The task identifies the intended Hugging Face
pipeline, while Mobius `text_only` maps the supported multimodal checkpoint to
the standalone `qwen3_5_moe_text` causal language model; the recipes do not rely
on task selection alone.

Mobius `text_only` is distinct from Olive's `components_to_export`: `text_only`
selects the text architecture during the Mobius build, while
`components_to_export` filters components from an already-built package while
saving. These recipes deliberately do not set `components_to_export`, because
doing so would suppress generation of the ONNX Runtime GenAI configuration.

Consequently, each output is a root-level decoder package containing
`model.onnx`, `genai_config.json`, and tokenizer assets rather than a `text/`
component inside a multimodal package. No vision encoder, image embedding
model, image processor, or other multimodal component is exported.

## Export methods

`MobiusBuilder` exports floating-point tensors and activations at fp16
precision for all four variants:

| Variant | Weight preparation |
|---|---|
| FP16 | No quantization pass; Mobius-only baseline |
| KQuant | MoE-aware symmetric int4, group size 128 |
| RTN | Round-to-nearest symmetric int4, group size 128, with MoE support |
| GPTQ | Calibration-aware symmetric int4, group size 128, with MoE support |

The target `LocalSystem` explicitly selects `CUDAExecutionProvider`, allowing
Mobius to build the graph and runtime configuration for CUDA. FP16 is the
precision validated for the complete KQuant CUDA graph. That measured artifact
uses standard `Conv` nodes after Mobius inlining; bf16 was not validated for
the complete graph.

KQuant and RTN both set `embeds: false` and `lm_head: false`. For KQuant, the
token embeddings and language-model head are therefore excluded explicitly.
MoE routers, `shared_expert_gate`, and the GatedDeltaNet `linear_attn`
recurrent/SSM blocks also remain floating point rather than becoming KQuant
targets.

The GPTQ pass receives `bits: 4`, `group_size: 128`, `sym: true`, `moe: true`,
and `lm_head: false`. Its pinned Olive schema does not support the `embeds`
option, so that option is intentionally not present in the GPTQ configuration.

KQuant and RTN do not require calibration data. GPTQ exactly follows the
comparison recipe's WikiText-2 policy: it loads
`Salesforce/wikitext` / `wikitext-2-raw-v1` `train`, joins text without adding
special tokens, and uses up to 512 samples of 2048 tokens with dataloader batch
size 1. The dataset is pinned to revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. GPTQ quality depends on
calibration coverage. Experts that receive insufficient routed calibration
tokens fall back to RTN; retain the fallback and unseen-expert counts from the
GPTQ export log with future benchmark results.

## Setup and export

Run from the `Qwen-Qwen3.6-35B-A3B/` directory:

```bash
pip install -r cuda/requirements.txt

olive run --config cuda/text/fp16/config.json
olive run --config cuda/text/kquant_fp16/config.json
olive run --config cuda/text/rtn_fp16/config.json
olive run --config cuda/text/gptq_fp16/config.json
```

Each workflow has an independent output:

| Variant | Configuration | Output |
|---|---|---|
| FP16 | `cuda/text/fp16/config.json` | `cuda/text/fp16/models/` |
| KQuant | `cuda/text/kquant_fp16/config.json` | `cuda/text/kquant_fp16/models/` |
| RTN | `cuda/text/rtn_fp16/config.json` | `cuda/text/rtn_fp16/models/` |
| GPTQ | `cuda/text/gptq_fp16/config.json` | `cuda/text/gptq_fp16/models/` |

The model source is pinned to revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. The dependency revisions preserve
Olive's MoE-aware quantization and MobiusBuilder behavior. Mobius PR #525
provides packed Olive QMoE sidecar handling, while Mobius PR #631 is required
for correct generic-decoder versus multimodal runtime routing.

## Runtime setup and compatibility

`requirements.txt` preserves the export dependencies and pins `lm-eval`
0.4.12 for the optional benchmark workflow. It does not install or replace
the source-built ONNX Runtime GenAI runtime used for this validation.

That result requires ONNX Runtime GenAI built from source at commit
`a8e0fdf81b061e67c1c3f9485bfdc06735ccd473` together with the
`onnxruntime-gpu` 1.30.0.dev20260823001 nightly built from ONNX Runtime commit
`4d308dacbb`. Follow the upstream
[ONNX Runtime GenAI build instructions](https://github.com/microsoft/onnxruntime-genai/blob/main/BUILD.md)
for the source build.

Stable ONNX Runtime GenAI 0.15.2 was not validated for this graph. The
generated `runtime_compatibility.json` values (minimum 0.14.0 and tested
0.15.2) are generic decoder ABI metadata and do not represent model-specific
runtime validation.

PR #588 remains a draft while Mobius PR #631 and a supported compatible
runtime distribution remain blockers. Until both are available, the source
revisions above are required to reproduce the validated KQuant runtime setup.

## Text inference

Use ONNX Runtime GenAI's upstream
[`model-qa.py`](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py)
sample. For example, from the recipe directory:

```bash
export ORT_GENAI_ROOT=/path/to/onnxruntime-genai
python "$ORT_GENAI_ROOT/examples/python/model-qa.py" \
  -m cuda/text/kquant_fp16/models/ \
  --user_prompt "Calculate 17 * 23 and state the final answer." \
  --max_length 512 \
  --non_interactive
```

Change `-m` to `cuda/text/fp16/models/`, `cuda/text/rtn_fp16/models/`, or
`cuda/text/gptq_fp16/models/` to test another variant. Use identical prompts,
generation options, runtime builds, and hardware for comparison.

The default chat template enables reasoning mode. Allow enough `max_length`
for reasoning prompts so generation can reach the final answer.

## Text evaluation

`cuda/text/eval/mmlu_cuda.json` declaratively evaluates the standalone ORT
GenAI package with Olive's `LMEvaluator` and `lm-eval`. It defaults to the
KQuant output and uses MMLU, the `ortgenai` model class, batch size 1, maximum
length 4096, limit 200, task-default few-shot behavior (`num_fewshot: null`),
no chat-template wrapping, and the CUDA execution provider.

Run from the `Qwen-Qwen3.6-35B-A3B/` directory:

```bash
# KQuant (the config default)
olive run --config cuda/text/eval/mmlu_cuda.json

# The same declarative evaluator settings for every variant
for model_path in \
  cuda/text/fp16/models \
  cuda/text/kquant_fp16/models \
  cuda/text/rtn_fp16/models \
  cuda/text/gptq_fp16/models
do
  olive run --config cuda/text/eval/mmlu_cuda.json \
    --model_name_or_path "$model_path"
done
```

For optional ad-hoc runs, use the same benchmark arguments for all four model
paths:

```bash
for model_path in \
  cuda/text/fp16/models \
  cuda/text/kquant_fp16/models \
  cuda/text/rtn_fp16/models \
  cuda/text/gptq_fp16/models
do
  olive benchmark \
    --model_name_or_path "$model_path" \
    --tasks mmlu \
    --device gpu \
    --backend ortgenai \
    --batch_size 1 \
    --max_length 4096 \
    --limit 200
done
```

The benchmark command's task-selected few-shot and chat-wrapping defaults match
the declarative settings above. Keep every evaluator option identical when
comparing variants. This is a text `lm-eval` path, not a VLM benchmark.

## Measured KQuant validation

The complete KQuant recipe was validated on 2026-08-25 with one NVIDIA
A100-SXM4-80GB and the CUDA execution provider:

- Olive loaded `Qwen/Qwen3.6-35B-A3B` at the pinned revision beginning
  `995ad96e` as `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text` with Transformers
  5.15.1 and the `text-generation` task.
- KQuant used symmetric int4 quantization with group size 128 and MoE support
  enabled, while excluding embeddings and the language-model head. It
  quantized 240 parameter tensors in 500.967169 seconds.
- The fp16 MobiusBuilder export completed in 143.995098 seconds.
- The root-level package is 19.53 GiB; `model.onnx.data` is
  20,948,779,008 bytes.
- `genai_config.json` identifies a `decoder` model with root-level
  `model.onnx`, enables CUDA graphs with `enable_cuda_graph=1`, and sets a
  262144-token context length.
- The generated `runtime_compatibility.json` records generic-decoder minimum
  ONNX Runtime GenAI 0.14.0 and tested version 0.15.2. Model-specific runtime
  validation instead used source-built ONNX Runtime GenAI 0.16.0.dev0 at
  commit `a8e0fdf81b061e67c1c3f9485bfdc06735ccd473` and
  `onnxruntime-gpu` 1.30.0.dev20260823001 from commit `4d308dacbb`.
- Exactly two reasoning-mode smoke runs used `model-qa.py` with greedy
  decoding (`top_k=1`). Both produced coherent reasoning that derived `391`
  for `Calculate 17 * 23 and state the final answer.`, but the capped runs
  were not evidence of completed final-channel answers.
- In a separate no-think check, the Hugging Face chat template was rendered
  with `enable_thinking=false`; generation produced the exact output `391`.
  This check was repeated successfully in the validated `qwen35-4b`
  environment.

The validated root-level standalone text package had the following layout.
The moved recipe writes the same package layout at its new output path:

```text
cuda/text/kquant_fp16/models/
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

The two reasoning-mode runs produced the following smoke measurements on that
one A100. These are not a broad benchmark:

| Max total tokens (prompt + generated cap) | Time to first token | Decode throughput |
|---:|---:|---:|
| 256 | 4.25 s | 77.66 tokens/s |
| 512 | 3.36 s | 93.16 tokens/s |

## Four-way benchmark results

FP16, RTN, and GPTQ were exported on 2026-08-26. KQuant reuses the validated
artifact exported on 2026-08-25 and documented above; its quantization pass
had already materialized `qwen3_5_moe_text` before MobiusBuilder, while the
moved recipe now also makes that selection explicit with `text_only: true`.
All four artifacts were evaluated on 2026-08-26 with one
NVIDIA A100-SXM4-80GB per run. The runtime and dependency revisions are the
ones documented above, including `lm-eval` 0.4.12. MMLU used all 57 subtasks,
up to 200 samples per subtask, task-default zero-shot prompting, no chat
template, batch size 1, and maximum length 4096. Subtasks with fewer than 200
test examples produced 9,183 effective samples in total.

| Variant | Package size | Export duration | MMLU accuracy | Difference from FP16 | Evaluation duration |
|---|---:|---:|---:|---:|---:|
| FP16 | 64.68 GiB | 23 min 16 s | **85.53% ± 0.35%** | -- | 2 h 15 min 54 s |
| KQuant INT4 | 19.53 GiB | 10 min 45 s | **84.51% ± 0.36%** | -1.01 pp | 10 min 42 s |
| RTN INT4 | 19.53 GiB | 4 min 51 s | **84.04% ± 0.37%** | -1.49 pp | 10 min 40 s |
| GPTQ INT4 | 19.53 GiB | 2 h 38 min 11 s | **84.33% ± 0.36%** | -1.20 pp | 10 min 34 s |

Export duration is the sum of the recorded Olive pass durations. It excludes
final output copying. For GPTQ, this includes 2 h 35 min 47 s for calibration
and quantization plus 2 min 24 s for MobiusBuilder.

The `Calculate 17 * 23 and state the final answer.` reasoning-mode smoke prompt
produced the correct value `391` for every package. The KQuant row reuses its
2026-08-25 validation run; the other three rows are from 2026-08-26. These
single runs are functional and throughput checks rather than a latency
benchmark:

| Variant | Max total tokens | Time to first token | Decode throughput |
|---|---:|---:|---:|
| FP16 | 256 | 2.21 s | 2.65 tokens/s |
| KQuant INT4 | 256 | 4.25 s | 77.66 tokens/s |
| RTN INT4 | 256 | 3.19 s | 78.25 tokens/s |
| GPTQ INT4 | 256 | 3.09 s | 78.36 tokens/s |

The GPTQ row above is the original 128-sample baseline. Its calibration used
WikiText-2 revision `b08601e` (dataset fingerprint
`5d4fb603254a7a5b`) and routed 262,144 tokens through each of the 40 MoE
layers. Across 10,240 layer-experts, 4,872 were starved and 3 were unseen.
Those 4,875 gate/up expert blocks used the RTN fallback instead of GPTQ. The
lower-dimensional down projections had sufficient coverage more often, with
2,212 of 10,240 expert blocks using the fallback. These fallback counts are
expected from the configured coverage thresholds and are part of the measured
GPTQ result rather than export failures.

The checked-in GPTQ recipe now uses 512 samples following a bounded
128→256→512 scaling study with identical model, quantization, preprocessing,
and MMLU settings. Every generation smoke test produced `391`. MMLU changes
are smaller than the combined standard errors, while gate/up fallback coverage
improves materially at both steps:

| Calibration blocks | Input tokens | Gate/up fallback | Down fallback | MMLU accuracy | Export pass duration |
|---:|---:|---:|---:|---:|---:|
| 128 | 262,144 | 4,875/10,240 (47.61%) | 2,212/10,240 (21.60%) | 84.33% ± 0.36% | 2 h 38 min 11 s |
| 256 | 524,288 | 2,847/10,240 (27.80%) | 2,024/10,240 (19.77%) | 84.69% ± 0.36% | 3 h 28 min 20 s |
| 512 | 1,048,576 | 2,059/10,240 (20.11%) | 2,059/10,240 (20.11%) | 84.58% ± 0.36% | 4 h 29 min 31 s |

The 512 result is selected because it passes the smoke and MMLU quality gates
and reduces gate/up fallback by another 7.70 percentage points from 256. The
paired per-example comparison over the same 9,183 MMLU examples found no
statistically significant accuracy difference:

| Comparison | Accuracy delta | Paired 95% confidence interval |
|---|---:|---:|
| 128 → 256 | +0.36 pp | [-0.07, +0.79] pp |
| 256 → 512 | -0.11 pp | [-0.56, +0.34] pp |

The selection therefore maximizes measured calibration coverage under the
bounded study; it does not establish 512 samples as a unique MMLU optimum. The
current ORT GenAI lm-eval adapter does not implement rolling loglikelihood, so
comparable WikiText test perplexity was unavailable and no proxy was used.

## References

- [Olive KQuant](https://github.com/microsoft/Olive/tree/main/olive/passes/pytorch/kquant.py)
- [Olive RTN](https://github.com/microsoft/Olive/tree/main/olive/passes/pytorch/rtn.py)
- [Olive GPTQ](https://github.com/microsoft/Olive/tree/main/olive/passes/pytorch/gptq.py)
- [Olive MobiusBuilder](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
- [Mobius](https://github.com/onnxruntime/mobius)
- [Mobius PR #525: packed Olive QMoE sidecars](https://github.com/onnxruntime/mobius/pull/525)
- [Mobius PR #631: standalone text and multimodal runtime type routing](https://github.com/onnxruntime/mobius/pull/631)
