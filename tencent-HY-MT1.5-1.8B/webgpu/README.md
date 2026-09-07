# tencent-HY-MT1.5-1.8B - WebGPU Optimization

This folder contains Olive recipes for optimizing `tencent/HY-MT1.5-1.8B` for `WebGpuExecutionProvider`.

## Recipes

- `tencent-HY-MT1.5-1.8B_webgpu_fp16.json`
- `tencent-HY-MT1.5-1.8B_webgpu_fp16_with_eval.json`
- `tencent-HY-MT1.5-1.8B_webgpu_int4.json`
- `tencent-HY-MT1.5-1.8B_webgpu_int4_with_eval.json`

## Setup

```bash
pip install -r requirements.txt
```

## Build examples

```bash
olive run --config tencent-HY-MT1.5-1.8B_webgpu_fp16.json
olive run --config tencent-HY-MT1.5-1.8B_webgpu_int4.json
```

## Build and evaluate with WMT18 Chinese-to-English translation

```bash
olive run --config tencent-HY-MT1.5-1.8B_webgpu_fp16_with_eval.json
olive run --config tencent-HY-MT1.5-1.8B_webgpu_int4_with_eval.json
```

## Notes

- HY-MT1.5-1.8B config has tie_word_embeddings=true, so TieWordEmbeddings surgery is applied after ModelBuilder.
- Full precision recipe for this backend uses `fp16`.
- Primary INT4 recipes use the full INT4 pass chain: GPTQ -> RTN -> ModelBuilder.
- Primary INT4 recipes omit SelectiveMixedPrecision and quantize embedding / lm_head to INT4 as well, while preserving WebGPU `group_size: 32`.
- The SelectiveMixedPrecision `ratio: 0.98` tuning output is kept as a higher-quality reference point, but it is no longer the primary INT4 recipe.

## Evaluation results

WMT18 Chinese-to-English translation, 3,981 test samples. BLEU and chrF are higher-is-better; TER is lower-is-better. The WebGPU-exported models were evaluated with ORT GenAI using CUDA EP for measurement.

| Model | Embedding / lm_head | Size | BLEU | chrF | TER |
| --- | --- | ---: | ---: | ---: | ---: |
| PyTorch baseline | fp16 | 3.806125 GiB | 12.634090637487896 | 35.290981008260474 | 85.7224234039956 |
| WebGPU INT4 (full INT4, primary) | int4 | 1.036270 GiB | 17.170642492358855 | 43.50620755758628 | 80.59466167555031 |
| WebGPU INT4 SMP (`ratio: 0.98`) | int8 | 1.175239 GiB | 19.914400461490622 | 51.365618622203314 | 78.65382267974122 |

## SelectiveMixedPrecision ratio tuning

The WebGPU sweep keeps `group_size: 32` and evaluates WebGPU-exported models with ORT GenAI using CUDA EP for measurement.

| Configured ratio | Size | BLEU | chrF | TER | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.83 | 1.303161 GiB | 19.361209647557704 | 47.93900531979513 | 81.56406396224112 | Earlier conservative point |
| 0.90 | 1.243736 GiB | 17.294982888634184 | 44.015683477618495 | 83.01664157545673 | Smaller, quality lower |
| 0.95 | 1.198408 GiB | 19.59367777635586 | 47.99027966955557 | 80.96187492370916 | Good size / quality point |
| 0.98 | 1.175239 GiB | 19.914400461490622 | 51.365618622203314 | 78.65382267974122 | Best tested SMP quality point |
| 0.985 | 1.170202 GiB | 15.526458652910305 | 41.07165704340729 | 83.61679619156122 | Quality cliff starts above 0.98 |
| 0.99 | 1.164157 GiB | 18.187055216810382 | 45.06622281076286 | 81.91093298612525 | Smaller, below 0.98 quality |

`ratio: 0.98` is the best tested SMP point: it is smaller than `0.83`, higher quality than `0.95`, and avoids the quality drop seen at `0.985` and `0.99`. The primary INT4 recipe now uses full INT4 because it is smaller than the SMP variants while still beating the PyTorch baseline on BLEU, chrF, and TER.
