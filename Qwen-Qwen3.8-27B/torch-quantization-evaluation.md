# Qwen3.8-27B native PyTorch quantization evaluation

These October 9-10, 2026 experiments compare Olive native `Rtn` and `KQuant`
checkpoints for `Qwen/Qwen3.8-27B`. They are **PyTorch quality diagnostics**, not
results for the CUDA or WebGPU ONNX recipes in this directory.

## Controlled setup

| Setting | Value |
|---|---|
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Architecture | `Qwen3_5ForConditionalGeneration`, task `image-text-to-text` |
| Source / inference dtype | BF16 source weights / FP16 inference |
| Quantization | INT3 or INT4, symmetric, group size 128 |
| Targets | 496 decoder linear weights; 24,350,556,160 elements |
| Excluded weights | Vision tower, token embeddings, language-model head |
| Hardware | One NVIDIA A100-SXM4-80GB |
| Generation | Greedy, cache enabled, `enable_thinking=False`, 16 new tokens |
| Image processing | Pixel-area bounds: 65,536 to 589,824; not side lengths |

RTN and KQuant were independently run on the same original BF16 source
checkpoint. INT4 was not produced by requantizing INT3. FP16 inference loaded
the saved quantized checkpoints without recalibration or requantization.
Packed weights remained unchanged, and scales matched the explicit BF16 to
FP16 cast. QuantTensor parameter/buffer references were refreshed after model
device moves.

The pass configuration was identical except for `type` and `bits`:

```json
{
    "type": "KQuant",
    "bits": 3,
    "group_size": 128,
    "sym": true,
    "lm_head": false,
    "quantize_vision": false,
    "moe": false
}
```

Use `Rtn` for RTN, or `bits: 4` for INT4. The input model's attributes selected
the decoder component with source paths `model.language_model.layers`,
`model.language_model.norm`, `model.language_model.embed_tokens`, and `lm_head`;
the embedding and head were excluded from quantization.

### Dataset and scoring

AI2D used `lmms-lab/ai2d`, test split, revision
`c83a9b9692933aff8349157c88a413df9d02c4e5`. A fixed random subset of 256
questions from the 3,088-question split was selected with seed `20261008`.
Every comparison reused the saved manifest, images, option order, prompts,
gold labels, and processor tensor fingerprints.

Prompts requested a direct option letter. The reported **explicit-answer**
parser was anchored at the beginning of the response: it accepted direct
letters and explicit answer prefixes, including Markdown and English/Chinese
prefixes, but did not search explanations for arbitrary option letters.
"Invalid" means no answer was parsed, not a runtime failure. A stricter parser
was retained separately.

These are subset measurements, not official full-split benchmark scores.
The experiment drivers, exact prompts/parser source, selected manifests,
predictions, and tensor diagnostics were retained in the experiment archive,
but are not included in this documentation-only change. Seeds and settings
alone do not constitute an exact reproduction package.

## AI2D results

All models below used FP16 inference and the same 16-token output limit.

| Checkpoint | Explicit correct / 256 | Accuracy | Change from original | Invalid |
|---|---:|---:|---:|---:|
| Original | 218 | 85.15625% | -- | 3 |
| RTN INT3 | 208 | 81.25000% | -3.90625 pp | 7 |
| KQuant INT3 | 197 | 76.953125% | -8.203125 pp | 17 |
| RTN INT4 | 212 | 82.81250% | -2.34375 pp | 11 |
| KQuant INT4 | 217 | 84.765625% | -0.390625 pp | 0 |

KQuant INT3's stricter score was 178 correct with 44 invalid responses; the
explicit parser gave the 197/17 result above. Both parsers agreed for the
other rows of the table.

RTN was better at INT3, whereas KQuant was better at INT4 in this setup.
KQuant INT4 generated one answer letter followed by EOS for every question
(two generated tokens). Its near-baseline total does not imply per-question
parity: 12 original-correct questions became wrong and 11 original-wrong
questions became correct.

Each INT3 checkpoint occupied about 14.48 GiB, versus 17.31 GiB for INT4,
including the nonquantized weights. These sizes are not ONNX package sizes
or inference-memory requirements.

## Paired failures and output-length diagnostic

KQuant had 29 INT4-correct/INT3-wrong questions and nine differences in the
opposite direction. Of the 29, 14 were invalid-format failures and 15 were
parsed wrong choices. The original model answered 23 of the 29 correctly;
both the original and RTN INT3 answered 18 correctly.

A separate fixed random 32-question pilot (seed `20261009`) increased the
generation limit to 64 tokens for the original, RTN INT3, and KQuant INT3.
Their same-row correct counts stayed at 26, 25, and 25 respectively. KQuant
still had three invalid responses, six hit the 64-token cap, and inspection
found no clear late explicit answer recovery. The original and RTN ended
after two tokens on every pilot question. This pilot does not establish that
longer generation never helps on other questions.

## Same-input activation diagnostics

Twelve cases were selected for diagnosis: four invalid-format failures, four
wrong-choice failures, two reverse differences, and two stable-correct
controls. This was a stratified diagnostic selection, not a random accuracy
sample. Each model received the identical multimodal prompt, followed by a
second forward pass conditioned on the **original model's answer letter**.

The saved generation's first token was reproduced in all 48 model/case
prefills. Some KQuant INT3 failures already diverged at that first token:

| AI2D row | Question (abridged) | Original / RTN INT3 / KQuant INT4 | KQuant INT3 |
|---:|---|---|---|
| 561 | Which label refers to the full moon? | `C` | `Based`; `C` still led among A-D |
| 856 | Which organism is most affected if seaweed disappears? | `C` | `Based`; `C` still led among A-D |
| 319 | If all trout are fished, what happens to mayflies? | `A` | Wrong letter `B` |

After the forced original letter, EOS was the highest-logit next token in
every diagnostic run. This is conditional evidence only: forcing a correct
letter does not prove the model would independently recover the answer.

Last-position outputs from all 64 decoder layers were recorded. Eighteen
linear projections across layers 0, 3, 31, 32, 60, and 63 were also evaluated
in isolation on identical original-model inputs.

| Prefill diagnostic aggregate | RTN INT3 | KQuant INT3 | KQuant INT4 |
|---|---:|---:|---:|
| Median of per-case mean isolated-linear relative L2 error | 0.173291 | 0.158401 | 0.078078 |
| Median last-decoder-layer relative L2 error | 0.744520 | 0.904647 | 0.421811 |

KQuant INT3 had lower isolated-linear error than RTN INT3 in 192/216 prefill
comparisons and 199/216 after the forced letter. Nevertheless, its median
last-layer prefill error was larger. Local reconstruction improvements do
not imply better end-to-end answers. These measurements do not identify a
causal faulty layer: full-layer outputs follow each model's own trajectory,
and only last-position activations were recorded.

## Numerical search diagnostics

A fixed sample covered 18 matrices and 1,152 groups (147,456 weights) per
method. This is not an all-weight audit.

KQuant INT3's actual FP16 reconstruction error relative to RTN INT3 was
about 0.7964 for unweighted squared error and 0.8119 for KQuant's weighted
objective. Thus lower sampled weight error coexisted with worse AI2D quality.
The objective uses weight-derived importance, not task activations.

Tracing the actual FP32 search on these groups found no violations of either
bound: re-rounding at the selected positive scale did not worsen the searched
weighted objective, and the searched objective did not exceed the FP32 RTN
reference. This verifies those bounds on this sample, not full algorithm
correctness across all dtypes or inputs.

### Candidate-span limitation

| Diagnostic | INT3 | INT4 |
|---|---:|---:|
| Default candidate factor range | 2.1-3.9 | 6.1-7.9 |
| Groups selecting the uppermost factor | 854/1,152 | 529/1,152 |
| Wider diagnostic range | 2.1-6.0 | 6.1-10.0 |
| Wider / default actual weighted reconstruction error | 0.94618 | 0.96554 |
| Groups improved / worsened after final packing | 823 / 22 | 522 / 29 |

The wider ranges were CPU numerical counterfactuals on the sampled weights,
**not new checkpoints or task-level experiments**. They show that the default
search boundary can limit reconstruction. They do not establish that widening
improves AI2D, that the current implementation is incorrect, or that it must
match a particular GGML implementation.

### Dtype and artifact checks

Actual BF16 packing differed from ideal FP32 rounding at the same stored
scales. The aggregate weighted-error penalty was about 0.22% for KQuant INT3,
0.94% for KQuant INT4, and 0.29% for RTN INT3. Per-group deviations existed;
these small aggregate values are not a bound on task sensitivity. CPU BF16
repacking exactly reproduced saved CUDA packed codes in all sampled groups.

All 333 vision tensors (921,460,192 bytes) retained identical keys, shapes,
dtypes, and hashes in the original and RTN/KQuant checkpoints. The saved
quantized target sets and FP16 casts were verified, and generation caches
contained 48 finite FP32 recurrent states. No artifact corruption was
observed in these checks.

## Provenance and remaining questions

The experiments used a frozen Olive working-tree snapshot based on commit
`08ba18176bdf625471a67bdb0411e28e6ba2f95c`, **with local changes**, not a pristine
release or that commit alone. Its 345 Python files had tree SHA256
`daf0d9bfdac286c0472961baab84f6cda2962d6a3d456266fc469b66efe0e5a0`.
Runtime versions were Python 3.12.13, PyTorch 2.13.0+cu130, Transformers 5.15.0,
Olive 0.11.0.dev0, Datasets 4.8.5, Accelerate 1.13.0, and Pillow 12.2.0.

Remaining questions include whether a wider search improves task quality,
which error directions or cross-layer interactions change answer behavior,
and how representative the sampled matrices are of all 496 targets.
Comparing more tasks and independent questions is necessary before making
general algorithm recommendations. Do not tune on these diagnostic questions
and then present them as independent validation.

No ONNX export/parity, acceleration, full AI2D split, or new INT4 text-PPL
qualification was performed. In particular, this record does not validate
the block-size-32, INT8-KV, speculative-decoding deployment recipe.
