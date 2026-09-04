# Qwen3.5-0.8B Vitis

| Component | Weights | Activations | Calibration |
|-----------|---------|-------------|-------------|
| `decoder` | UINT8 | UINT16 | WikiText-2 train, 128 samples, up to 512 tokens each |
| `vision_encoder` | Symmetric INT4 RTN, block 128 | FP32 | None |
| `embedding` | Unchanged FP32 | FP32 | None |

## 1. Install

```powershell
pip install -r requirements.txt
```

## 2. Export

```powershell
olive capture-onnx-graph --model_name_or_path Qwen/Qwen3.5-0.8B --use_mobius_builder --precision fp32 --output_path exported_model
```

The full FP32 export includes `decoder`, `vision_encoder`, and `embedding`.
Keep the embedding component: text calibration uses it to generate decoder inputs.

## 3. Quantize

```powershell
olive run --config config.json
```

`config.json` applies separate pipelines through `builds.components`:

| Component | Pipeline |
|-----------|----------|
| `decoder` | `gs` -> `sq` -> `text_metadata` |
| `vision_encoder` | `gemm_to_matmul` -> `vision_rtn` -> `mq` -> `vision_metadata` |

Output:

```text
optimized_model/
  decoder/model.onnx
  vision_encoder/model.onnx
```

External weight files accompany each graph. The unchanged embedding component,
`genai_config.json`, tokenizer and processor files remain in `exported_model`.

## Calibration and precision

`wikitext2_train` uses `Salesforce/wikitext`, subset `wikitext-2-raw-v1`, with
line-by-line tokenization and no added special tokens. Adjust `max_samples` and
`max_seq_len` in `config.json` to change calibration size.

`user_script.py` supplies embeddings, MRoPE positions and hybrid KV/recurrent
states. It processes tokens sequentially and resets states between samples.

Text MatMul weights use UINT8; RoPE sin/cos caches use UINT16. Vision MatMul
weights use INT4 QDQ, while the patch-embedding Conv remains FP32.
Normalization operators excluded by `sq` remain floating point.
