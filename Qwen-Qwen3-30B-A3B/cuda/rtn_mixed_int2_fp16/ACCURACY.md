# Quality Evaluation

Measured October 4, 2026, for the same matched block64 exports described in
the recipe README. The WikiText-2 and MMLU checks used direct ORT with Release
`main@62ac19abcf` on A100-SXM4-80GB GPU 1 in Docker `jiafa-dev`.
ARC-Easy used Olive's ORT backend on GPU 1 with the same ORT build.
WikiText-2 and MMLU are small-sample diagnostics; ARC-Easy covers the full
test split. These results do not establish accuracy preservation.
No unquantized FP16/Hugging Face quality reference or
independent numerical parity check was evaluated.

## Full ARC-Easy Test

Olive `LMEvaluator` calls `lm_eval.simple_evaluate` using the standard
`arc_easy` task from lm-evaluation-harness 0.4.12. Dataset:
`allenai/ai2_arc`, configuration `ARC-Easy`, all 2376 test documents.
Zero-shot, batch size 1, maximum input length 2048, no chat template.
The standard prompt is `Question: {question}\nAnswer:`; candidate answer
texts are scored by their continuation loglikelihood, not custom A/B/C tokens.
There was no input truncation; prefill input lengths were 8-166 tokens.

| Model | Correct (`acc`) | `acc` | Correct (`acc_norm`) | `acc_norm` |
|---|---:|---:|---:|---:|
| INT4 block64 | 1877 / 2376 | 78.9983% | 1858 / 2376 | 78.1987% |
| Mixed INT2/INT4 block64 | 1831 / 2376 | 77.0623% | 1740 / 2376 | 73.2323% |

Mixed decreases `acc` by **1.9360 percentage points** and `acc_norm` by
**4.9663 percentage points**. `acc` selects the candidate with the highest
total loglikelihood. In this lm-eval version, `acc_norm` divides that score
by the candidate string's character count (not token count) before selecting.
Both metrics are reported; they use different answer-selection rules.
The earlier 50-document pilot did not predict the full-test normalized-score
direction and should not be used to claim a quality improvement.

All paired questions, options, targets, tokenizer hashes and ORT builds were
checked for equality. Raw accuracy was independently recomputed from the
2376 per-model sample records: 150 correct-to-wrong changes and 104
wrong-to-correct changes. The shared document SHA256 is
`308cd49b506c74e78d496d7f23f5e2b830debc0cfe63ecd9bca0b53c8ef4ec74`.

### Packed INT2 Prefill Verification

Both full runs set `ORT_QMOE_INT_DEQUANT_MAX_SCRATCH_BYTES=1`.
Mixed additionally set `ORT_ENABLE_QMOE_KERNEL_DEBUG_INFO=1`. All 2376
documents completed under this guard, which rejects dense INT2 expert-weight
dequantization; it is not a total workspace or GPU-memory limit.
The complete mixed route log contains:

- `packed_int_prefill`: 195888 calls for the 24 INT2 gate/up layers.
- `grouped_moe`: 195888 calls for the remaining 24 INT4 expert layers.
- 8162 shared-context prefill calls after lm-eval's continuation grouping.

No other QMoE routes, dense-fallback errors or input truncations were found.
This verifies packed INT2 prefill for this workload, not arbitrary model
shapes, runtimes or generation/decode workloads.

The evaluated backend was Olive's ordinary `ort` adapter, not `ortgenai`.
A separate short-prompt check with the container's GenAI 0.18.0-dev entered
dense fallback and failed its scratch limit; that runtime is not qualified
by the successful direct-ORT results. Local Olive gained an optional
`hf_config_path` parameter so tokenizer/HF configuration assets can be read
separately while loading the original ONNX graph and external weights.
The neighboring evaluator tests passed (9 tests). This local compatibility
change is not included in this recipe PR.

## WikiText-2 Perplexity

Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, test split. All 4358 text
rows were joined with two newlines and tokenized without added special tokens,
BOS or chat template. The full text contains 299078 Qwen tokens. This check
uses only the first 10001 tokens, predicting exactly 10000 next-token targets
(about 3.3% of the full sequence).

Context length 1024, stride 512; empty KV reset for each window. Overlapping
windows score only newly exposed targets, so every target is counted once.
Teacher forcing uses logits at position t to score token t+1. Stable float64
log-sum-exp is applied to the original FP16 logits. PPL is the exponential of
total negative log likelihood divided by the scored-token count.

| Model | Mean NLL (nats/token) | Perplexity | Load (s) | Evaluation (s) |
|---|---:|---:|---:|---:|
| INT4 block64 | 2.238341 | 9.377757 | 62.93 | 22.90 |
| Mixed INT2/INT4 block64 | 2.331856 | 10.297031 | 47.66 | 20.67 |

Mixed PPL is **9.80% higher (worse)** and NLL increases by 0.093515 nats/token.
This suggests worse text prediction on this sample; it does not mean a
9.80-percentage-point decrease in question-answering accuracy. Full-test PPL
and task-level evaluation are needed before broader conclusions.

Both runs used identical dataset/tokenizer/token hashes and window schedules;
all scored logits were finite. Window coverage and analytic cross-entropy
self-tests passed. NLL totals and PPL were recomputed from per-window records.

## MMLU 50-Question Smoke Test

Dataset: `cais/mmlu`, configuration `all`, test split; dev split provides
five demonstrations per subject. Random seed 42 selects 50 of the 57 subjects,
then one random test question per selected subject. This is **50 questions
total**, not 50 per subject. The same question IDs and demonstrations are
used for both models.

The prompt follows the lm-eval default MMLU style: subject description,
five answered dev examples, question and A/B/C/D choices, ending in `Answer:`.
No chat template, thinking prompt or free-form generation is used.
Batch size is 1; prompt lengths are 252-2648 tokens, without truncation.

A custom direct-ORT harness scores the continuations ` A`, ` B`, ` C`, ` D`.
All four were verified to be single tokens with an unchanged context-token
prefix. One prefill therefore supplies all four answer log probabilities;
the highest score selects the answer. This is the single-token shared-context
likelihood optimization also used by the Olive lm-eval adapter, not four
independent generations. These results are not an official full lm-eval run.

| Model | Correct / Total | Accuracy | Load (s) | Evaluation (s) |
|---|---:|---:|---:|---:|
| INT4 block64 | 38 / 50 | 76% | 61.57 | 36.93 |
| Mixed INT2/INT4 block64 | 37 / 50 | 74% | 41.41 | 19.87 |

Mixed scored one fewer correct answer: a **2-percentage-point sample delta**.
Four predictions changed: one wrong-to-correct, two correct-to-wrong and one
wrong-to-different-wrong. This sample is too small to establish a stable
accuracy gap. Subject-balanced sampling does not reproduce the full MMLU
question weighting; scores must not be compared directly with the earlier
9183-question recipe results.

Per-question median times were 61.0 ms for INT4 and 54.9 ms for mixed, but
several calls had much larger cold/shape-dependent overheads. The total times
are not warmed performance measurements. Naive scaling of the 50-question
totals gives about 113/61 minutes respectively for 9183 questions, but that
repeats cold overhead and ignores prompt-length distributions. It is not a
reliable completion-time forecast. A larger warmed pilot is needed.

Both checks retained `ep.cuda.qmoe_int_dequant_max_scratch_bytes=1`, a guard
against full dense expert-weight materialization, not a total memory limit.
The runs completed with finite scored logits. Batch size 8 was not tested.

## Local Reproduction Artifacts

Harnesses under `/datadisks/disk1/jiafa/accuracy/`:

- `qwen3_wikitext_ppl.py`: window-coverage self-test, sample preparation and PPL.
- `qwen3_mmlu_smoke.py`: fixed 5-shot 50-question preparation and likelihood scoring.
- `qwen3_olive_lmeval.py`: standard Olive/lm-eval task evaluation using the ORT backend.

Artifacts under `qwen3-perf-block64-20261004/`:

- `wikitext2-sample-tokens.json`, `wikitext2-int4-sample.json`,
  `wikitext2-mixed-sample.json`, `WIKITEXT2_SAMPLE.md`.
- `mmlu-50-sample.json`, `mmlu-int4-50.json`, `mmlu-mixed-50.json`.
- `olive-arc-easy-int4-full/results.json` and `samples/arc_easy_samples.jsonl`.
- `olive-arc-easy-mixed-full/results.json` and `samples/arc_easy_samples.jsonl`.
- `olive-arc-easy-mixed-full.log`: complete mixed kernel-route log.

Both MMLU result files reference sample SHA256
`f357ff5dddc9485253f07f7135f95615280f061b5faa24e0177a1d03574b95f1`.
The fixed samples, source dataset content and raw artifacts are retained
locally, not uploaded as part of this recipe.