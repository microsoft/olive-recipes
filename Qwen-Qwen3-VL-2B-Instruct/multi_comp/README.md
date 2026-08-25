# Qwen3-VL-2B-Instruct — Multi-Component Optimization

These recipes demonstrate two multi-component flows for
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct):

- **Flow A — export first, then per-component optimization**
  (`vlm_optimize_components.json`): export the VLM to ONNX once with the Mobius builder, then run a
  single Olive config whose `builds` apply a **different pipeline to each component**.
- **Flow B — optimize a Torch component first, then export**
  (`vlm_quantize_then_export.json`): run a Torch-stage GPTQ pass on the decoder component while
  saving a complete HF directory, then export that directory with
  `olive capture-onnx-graph --use_mobius_builder`.

Olive loads an exported directory as a `CompositeModel` whose **component names are the subfolder
names**, so there is no need to memorize component names.

## Prerequisites

```
pip install olive-ai
pip install mobius-ai
```

Exporting also needs `transformers` and access to the model on Hugging Face.

---

## Recipe 1 — Export then per-component optimize (`vlm_optimize_components.json`)

### Step 1 — Export

```
olive capture-onnx-graph --model_name_or_path Qwen/Qwen3-VL-2B-Instruct --use_mobius_builder --output_path exported_vlm_pkg
```

Mobius exports this model as three components, each in its own subfolder:

```
exported_vlm_pkg/
  decoder/model.onnx
  vision_encoder/model.onnx
  embedding/model.onnx
```

### Step 2 — Optimize

```
olive run --config vlm_optimize_components.json
```

| component        | pipeline        | intent                              |
|------------------|-----------------|-------------------------------------|
| `decoder`        | `dynamic_quant` | INT8-quantize the language decoder  |
| `vision_encoder` | `to_fp16`       | keep the vision tower in FP16       |
| `embedding`      | `to_fp16`       | keep the embedding in FP16          |

> The three component names (`decoder`, `vision_encoder`, `embedding`) are exactly what Mobius
> produces for `Qwen/Qwen3-VL-2B-Instruct`. For a different VLM, adjust the component names in the
> config to match the subfolder names your export actually produced.

### Step 3 — Inference with ORT GenAI

Run text generation with the exported ONNX models using **onnxruntime-genai**:

```bash
# Text-only
python vlm_inference.py --prompt "The capital of France is"

# With image input
python vlm_inference.py --prompt "Describe this image." --image photo.jpg

# Custom settings
python vlm_inference.py --model_dir exported_vlm_pkg --max_new_tokens 256
```

The inference script (`vlm_inference.py`) uses ORT GenAI which handles:
- **Tokenization**: built-in tokenizer from saved HF tokenizer files
- **Embedding**: ONNX `embedding/model.onnx` (token embed + image feature mixing)
- **Vision encoding**: ONNX `vision_encoder/model.onnx` (when `--image` is provided)
- **Decoding**: ONNX `decoder/model.onnx` with KV cache (autoregressive generation)

Options:
```
--prompt TEXT           Text prompt
--image PATH            Optional image file for multimodal input
--max_new_tokens N      Maximum tokens to generate (default: 128)
--model_dir DIR         Path to exported model directory (default: exported_vlm_pkg)
```

#### Setup requirements

The export directory needs these files alongside the ONNX models:

```
exported_vlm_pkg/
  genai_config.json          # Model type, I/O mappings, search config
  tokenizer.json             # HF tokenizer
  tokenizer_config.json
  vision_processor.json      # Vision preprocessing config
  decoder/model.onnx
  vision_encoder/model.onnx
  embedding/model.onnx
```

To create the tokenizer files after export:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True)
tokenizer.save_pretrained("exported_vlm_pkg")
```

For the `genai_config.json` structure, see the
[Mobius ORT GenAI examples](https://github.com/microsoft/mobius/tree/main/examples) which write the
config automatically.

> **Note.** Install `onnxruntime-genai` (`pip install onnxruntime-genai`) to use this script.

---

## Recipe 2 — Torch decoder quantization, then Mobius export (`vlm_quantize_then_export.json`)

This is **Flow B**: quantize only the Torch decoder component first, then export the resulting
complete HF directory with the Olive capture CLI using the Mobius builder.

### Step 1 — Quantize the decoder component

```
olive run --config vlm_quantize_then_export.json
```

The config uses `builds.components: ["decoder"]`, so Olive asks Mobius for the VLM component plan,
scopes the Torch GPTQ pass to the decoder submodule, and saves the original HF folder layout with
the decoder quantized in place. This output is **not** a standalone decoder checkpoint; it is a
complete HF model directory:

```
out/vlm_decoder_gptq_hf/
```

The recipe uses the GPTQ pass defaults for calibration data (`Salesforce/wikitext`). For production,
add a `data_config` to `decoder_gptq` with your own text or multimodal calibration set.

### Step 2 — Export the quantized HF directory with the Mobius builder

```
olive capture-onnx-graph \
  --model_name_or_path vlm_decoder_gptq_hf \
  --use_mobius_builder \
  --trust_remote_code \
  --precision fp16 \
  --output_path exported_vlm_gptq_pkg
```

Output:

```
exported_vlm_gptq_pkg/
  decoder/model.onnx
  vision_encoder/model.onnx
  embedding/model.onnx
```

> **Note.** The Torch GPTQ pass saves Olive-packed weights (`quant_method="olive"`). Use this export
> step with a Mobius builder version that supports Olive-packed quantized HF checkpoints.

---

## Notes

- The passes in Recipe 1 (`OnnxFloatToFloat16`, `OnnxDynamicQuantization`) are **illustrative** and
  chosen to run without calibration data. Swap in `OrtTransformersOptimization`,
  `OnnxStaticQuantization` (with a `data_config`), or other ONNX passes for production-quality
  optimization.
- The ONNX component recipe runs on the EP declared in its `systems` section. The Torch GPTQ recipe
  targets CUDA because VLM decoder GPTQ is GPU-oriented.
- `builds.components` selects which exported components to optimize. Only the components with a build
  are touched; the rest remain as exported.
