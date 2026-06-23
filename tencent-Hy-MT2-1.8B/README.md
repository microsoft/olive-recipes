# Hy-MT2-1.8B (tencent/Hy-MT2-1.8B)

Olive recipes that export [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)
to ONNX via the [`MobiusBuilder`](https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py)
pass and (optionally) quantize the decoder with Olive's K-Quant (Q4_K_M)
pass for INT4 deployment.

Hy-MT2-1.8B is a "fast-thinking" multilingual **translation** model
supporting 33 languages. Architecturally it is a `HunYuanDenseV1` decoder:
grouped-query attention (16 query / 4 KV heads, head_dim 128), per-head
QK-norm, tied input/output embeddings, SwiGLU MLP, and dynamic-NTK RoPE
(`alpha = 1000`). `MobiusBuilder` builds the decoder directly with the fused
GroupQueryAttention path on CUDA.

`MobiusBuilder` writes a fully-formed ORT GenAI package (`genai_config.json`,
`tokenizer.json`, `chat_template.jinja`) alongside the ONNX files — no
post-processing required.

## Prerequisites

```bash
pip install -r requirements.txt
```

`requirements.txt` pulls in `olive-ai[gpu]`, `mobius-ai`, and `lm-eval`.

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| CPU | `pip install onnxruntime-genai` |
| GPU (CUDA) | `pip install onnxruntime-genai-cuda` |

## Recipes

| Recipe | Pipeline | Output dir |
|---|---|---|
| `cpu/fp32/config.json` | `MobiusBuilder(fp32)` | `cpu/fp32/models` |
| `cpu/int4/config.json` | `MobiusBuilder(fp32)` → `OnnxKQuantQuantization(bits=4, block_size=32)` | `cpu/int4/models` |
| `cuda/fp16/config.json` | `MobiusBuilder(fp16)` | `cuda/fp16/models` |
| `cuda/bf16/config.json` | `MobiusBuilder(bf16)` | `cuda/bf16/models` |
| `cuda/int4/config.json` | `MobiusBuilder(fp16)` → `OnnxKQuantQuantization(bits=4, block_size=32)` | `cuda/int4/models` |

INT4 (K-Quant Q4_K_M) is recommended for most deployments — at 1.8B
parameters it is a ~1 GB on-disk model with negligible impact on translation
quality. K-Quant is significantly faster with GPU acceleration — install
`cupy-cuda12x` for a large speedup during quantization.

> BF16 MatMul is not implemented on the ORT CPU EP, so the `bf16` variant
> runs on CUDA / DML / WebGPU only.

## Build

```bash
# CPU, full precision
olive run --config cpu/fp32/config.json

# CPU, INT4 (K-Quant)
olive run --config cpu/int4/config.json

# CUDA, FP16
olive run --config cuda/fp16/config.json

# CUDA, BF16
olive run --config cuda/bf16/config.json

# CUDA, INT4 (K-Quant)
olive run --config cuda/int4/config.json
```

Each command produces the full ORT GenAI package in the recipe's
`output_dir`:

```
<output_dir>/
├── model.onnx              # Decoder-only transformer
├── model.onnx.data         # External weights
├── genai_config.json       # Runtime configuration
├── chat_template.jinja
├── tokenizer.json
└── tokenizer_config.json
```

## Inference

```bash
# CPU, fp32
python inference.py --source "今天天气真好。" --target-lang English

# CPU INT4
python inference.py --variant int4 --source "Hello, world!" --target-lang Chinese

# CUDA INT4
python inference.py --device gpu --variant int4 --source "Knowledge is power." \
    --target-lang French

# CUDA BF16, interactive
python inference.py --device gpu --variant bf16 --interactive
```

The Hy-MT BPE vocab uses a custom regex pre-tokenizer that ort-extensions
does not currently round-trip, so `inference.py` tokenizes / detokenizes with
the HuggingFace tokenizer and feeds raw token IDs to `og.Generator` (this
still exercises the full ORT GenAI inference path).

## Verification

The mobius ONNX export is verified token-for-token against HuggingFace
`transformers` (greedy decode, L4 prefill logits + L5 generation) in the
[mobius repository](https://github.com/onnxruntime/mobius)
(`tests/e2e_golden_test.py`, case `causal-lm/hy-mt2-1_8b`). Example:

```
source : 黄河之水天上来
ONNX   : The waters of the Yellow River come from the sky
HF     : The waters of the Yellow River come from the sky
```

## References

- Hy-MT2 model card: <https://huggingface.co/tencent/Hy-MT2-1.8B>
- Mobius: <https://github.com/onnxruntime/mobius>
- Olive `MobiusBuilder` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/mobius_model_builder.py>
- Olive `OnnxKQuantQuantization` pass: <https://github.com/microsoft/Olive/tree/main/olive/passes/onnx/kquant_quantization.py>
