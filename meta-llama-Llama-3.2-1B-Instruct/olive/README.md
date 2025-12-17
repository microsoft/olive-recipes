## Llama 3.2 1B Instruct Recipes

This folder provides Olive optimization / fine-tuning / quantization / evaluation recipes for `meta-llama/Llama-3.2-1B-Instruct`.

Each recipe is a self‑contained JSON passed to the Olive CLI: `olive run --config <file>.json`.

### Quick Start

Install dependencies (make sure you are in the `olive-recipes/meta-llama-Llama-3.2-1B-Instruct/olive` directory):

```bash
python -m pip install -r requirements.txt
```

Typical steps:

### Run optimization / finetuning:
```bash
olive run --config qlora.json
```

Output models & adapters saved under the `output_dir` (default `models/`).

### Recipe Summary

| File | Goal | Main Pass Chain (order) |
|------|------|-------------------------|
| `qlora.json` | QLoRA PEFT finetune + export + ORT opt + extract adapters | q (qlora) → m (ModelBuilder fp16) → o (OrtTransformersOptimization fp16) → e (ExtractAdapters) |
| `loha.json` | LoHa finetune + ONNX export + ORT opt + extract | l (loha) → c (OnnxConversion) → o (ORT opt) → e (ExtractAdapters) → m (metadata) |
| `lokr.json` | LoKr finetune + ONNX export + ORT opt + extract | l (lokr) → c (OnnxConversion) → o → e → m |
| `dora.json` | DoRA finetune + ORT opt + extract | d (dora) → m (ModelBuilder fp16) → o → e |
| `rtn.json` | Block‑wise RTN quantization (ONNX) | m (ModelBuilder fp16) → q (OnnxBlockWiseRtnQuantization) |
| `hqq.json` | HQQ quantization (ONNX) | m (ModelBuilder fp16) → q (OnnxHqqQuantization) |
| `lmeval.json` | HF (fp16/fp32) evaluation with LMEval | evaluator only |
| `lmeval_onnx.json` | INT4 ModelBuilder + LMEval | mb (ModelBuilder int4) + evaluator |

### Example Commands

QLoRA training + optimization:
```bash
olive run --config qlora.json
```

LoHa adapter training and export to ONNX:
```bash
olive run --config loha.json
```

HQQ quantization (after ONNX build inside pass chain):
```bash
olive run --config hqq.json
```

Run LM evaluation on HF model:
```bash
olive run --config lmeval.json
```

Evaluate INT4 ONNX build:
```bash
olive run --config lmeval_onnx.json
```

Clean cache for a fresh run (example):
```bash
olive run --config qlora.json --clean_cache
```
