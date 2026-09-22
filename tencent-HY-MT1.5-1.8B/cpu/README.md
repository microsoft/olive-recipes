# tencent-HY-MT1.5-1.8B - CPU Optimization

This folder contains Olive recipes for optimizing `tencent/HY-MT1.5-1.8B` for `CPUExecutionProvider`.

## Recipes

- `tencent-HY-MT1.5-1.8B_cpu_fp32.json`
- `tencent-HY-MT1.5-1.8B_cpu_fp32_with_eval.json`
- `tencent-HY-MT1.5-1.8B_cpu_int4.json`
- `tencent-HY-MT1.5-1.8B_cpu_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config tencent-HY-MT1.5-1.8B_cpu_fp32.json
olive run --config tencent-HY-MT1.5-1.8B_cpu_int4.json
```

## Build and evaluate with WMT18 Chinese-to-English translation

```bash
olive run --config tencent-HY-MT1.5-1.8B_cpu_fp32_with_eval.json
olive run --config tencent-HY-MT1.5-1.8B_cpu_int4_with_eval.json
```

## Notes

- HY-MT1.5-1.8B config has tie_word_embeddings=true, so TieWordEmbeddings surgery is applied after ModelBuilder.
- Full precision recipe for this backend uses `fp32`.
- The primary INT4 recipe uses full INT4 with `group_size: 32`: GPTQ -> RTN -> ModelBuilder.
- The primary INT4 recipe omits SelectiveMixedPrecision and quantizes embedding / lm_head to INT4 as well.
- CPU recipes save ONNX weights to `model.onnx.data` through the final GraphSurgeries pass.
- Full INT4 with `group_size: 32` was selected because it is smaller than the previous CPU INT4 artifact and still improves WMT18 Chinese-to-English quality over the PyTorch baseline when evaluated with CPU EP.

## Evaluation results

WMT18 Chinese-to-English translation, 3,981 test samples. BLEU and chrF are higher-is-better; TER is lower-is-better. CPU ONNX results were evaluated with ORT GenAI using CPU EP.

| Model | Embedding / lm_head | Size | BLEU | chrF | TER |
| --- | --- | ---: | ---: | ---: | ---: |
| PyTorch baseline | fp16 | 3.806125 GiB | 12.634090637487896 | 35.290981008260474 | 85.7224234039956 |
| Current CPU INT4 recipe (full INT4, `group_size: 32`) | int4 | 1.203313 GiB | 16.920205475631327 | 42.982874679434815 | 80.8764291817553 |

The current INT4 artifact is 68.4% smaller than the PyTorch model and improves BLEU by 4.29, chrF by 7.69, and TER by 4.85 points. It is also smaller than the previous CPU INT4 artifact based on SelectiveMixedPrecision `ratio: 0.83` (1.288222 GiB), so the full INT4 `group_size: 32` recipe is the primary CPU INT4 path.
