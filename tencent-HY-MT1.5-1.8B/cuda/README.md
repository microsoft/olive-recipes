# tencent-HY-MT1.5-1.8B - CUDA Optimization

This folder contains Olive recipes for optimizing `tencent/HY-MT1.5-1.8B` for `CUDAExecutionProvider`.

## Recipes

- `tencent-HY-MT1.5-1.8B_cuda_fp16.json`
- `tencent-HY-MT1.5-1.8B_cuda_fp16_with_eval.json`
- `tencent-HY-MT1.5-1.8B_cuda_int4.json`
- `tencent-HY-MT1.5-1.8B_cuda_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config tencent-HY-MT1.5-1.8B_cuda_fp16.json
olive run --config tencent-HY-MT1.5-1.8B_cuda_int4.json
```

## Build and evaluate with WMT18 Chinese-to-English translation

```bash
olive run --config tencent-HY-MT1.5-1.8B_cuda_fp16_with_eval.json
olive run --config tencent-HY-MT1.5-1.8B_cuda_int4_with_eval.json
```

## Notes

- HY-MT1.5-1.8B config has tie_word_embeddings=true, so TieWordEmbeddings surgery is applied after ModelBuilder.
- Full precision recipe for this backend uses `fp16`.
- Primary INT4 recipes use full INT4 with `group_size: 32`: GPTQ -> RTN -> ModelBuilder.
- Primary INT4 recipes omit SelectiveMixedPrecision and quantize embedding / lm_head to INT4 as well.
- The previous SelectiveMixedPrecision `ratio: 0.83` primary remains in the tuning history, but it has been replaced by the smaller and higher-quality full INT4 `group_size: 32` recipe.

## Evaluation results

WMT18 Chinese-to-English translation, 3,981 test samples. BLEU and chrF are higher-is-better; TER is lower-is-better.

| Model | Embedding / lm_head | Size | BLEU | chrF | TER |
| --- | --- | ---: | ---: | ---: | ---: |
| PyTorch baseline | fp16 | 3.806125 GiB | 12.634090637487896 | 35.290981008260474 | 85.7224234039956 |
| Previous CUDA INT4 recipe (`ratio: 0.65`) | int8 | 1.350105 GiB | 15.265524905493395 | 39.79625126006993 | 84.66147210806851 |
| Previous CUDA INT4 recipe (`ratio: 0.83`) | int8 | 1.199468 GiB | 13.362969543291351 | 36.72419140068781 | 85.49761972575986 |
| Current CUDA INT4 recipe (full INT4, `group_size: 32`) | int4 | 1.036290 GiB | 17.170642492358855 | 43.50620755758628 | 80.59466167555031 |
| CUDA full INT4 (`group_size: 128`) | int4 | 0.938559 GiB | 3.759288935325742 | 20.2313593462662 | 93.06058509989013 |
| CUDA INT4 / INT8 mixed | int8 | 1.054681 GiB | 3.325815186575075 | 19.319673423679575 | 93.61699963380397 |

The current CUDA INT4 recipe uses full INT4 GPTQ for the body and full INT4 RTN for embedding / lm_head. All CUDA ONNX variants apply TieWordEmbeddings after ModelBuilder.

The previous `ratio: 0.83` primary CUDA INT4 output and cache were backed up before promoting full INT4 `group_size: 32` to `model_cuda_int4`.

## SelectiveMixedPrecision ratio tuning

`ratio` is the fraction of scored parameters kept at the default low precision. Higher values usually reduce model size, but after the sensitive layers are no longer promoted to 8-bit, translation quality drops quickly.

| Recipe | Configured ratio | Size | BLEU | chrF | TER | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Previous primary | 0.65 | 1.350105 GiB | 15.265524905493395 | 39.79625126006993 | 84.66147210806851 | Replaced |
| Tuning run | 0.75 | 1.268385 GiB | 17.78380603332943 | 44.55479318041093 | 82.10928917280384 | Best quality |
| Tuning run | 0.80 | 1.224083 GiB | 16.8883865898559 | 43.2185687820676 | 82.54465557228303 | Best quality / size point versus previous INT4 |
| Tuning run | 0.81 | 1.216207 GiB | 14.359839243852116 | 38.37412933939934 | 84.75810717337347 | Smaller, but no longer beats previous primary on all metrics |
| Tuning run | 0.82 | 1.211281 GiB | 13.304635612235579 | 36.437981160175106 | 85.74683647312527 | Near baseline, TER slightly below PyTorch |
| Previous primary | 0.83 | 1.199468 GiB | 13.362969543291351 | 36.72419140068781 | 85.49761972575986 | Replaced by full INT4 `group_size: 32` |
| Tuning run | 0.84 | 1.193562 GiB | 14.45227322167928 | 38.44629014667923 | 86.48940065915286 | Smaller and BLEU/chrF above PyTorch, but TER below PyTorch |
| Tuning run | 0.85 | 1.181749 GiB | 10.513690403174172 | 31.910545705126708 | 88.3702241933515 | Quality collapse |
| Tuning run | 0.90 | 1.140391 GiB | 6.483511887686124 | 25.198896559504853 | 91.26825894128658 | Quality collapse |

Compared with PyTorch, the current full INT4 `group_size: 32` recipe is 72.8% smaller and improves BLEU by 4.54, chrF by 8.22, and TER by 5.13 points. The SelectiveMixedPrecision sweep remains as historical tuning data, but it is no longer the recommended CUDA INT4 path.

The CUDA INT4 / INT8 mixed result was reproduced after clearing `cache_cuda_int4_int8` and `model_cuda_int4_int8`, then re-exporting and re-evaluating from scratch.
