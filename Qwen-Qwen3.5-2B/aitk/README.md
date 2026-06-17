# Qwen3.5-2B Model Optimization — NVIDIA TRT for RTX

This recipe converts the [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
vision-language model to ONNX for the **NVIDIA TensorRT for RTX** execution
provider (`NvTensorRTRTXExecutionProvider`) and runs it with ONNX Runtime
GenAI.

Qwen3.5 is a hybrid architecture combining GatedDeltaNet linear attention
layers with standard full attention layers. The pipeline exports three
sub-models and assembles them into a single ONNX Runtime GenAI model folder:

- **embedding.json** — token embedding + image feature fusion (FP16)
- **vision.json** — vision encoder, packed patches → image features (FP16)
- **text.json** — text decoder via ModelBuilder (INT4, hybrid GatedDeltaNet + full attention)

Because AITK runs a single Olive workflow per recipe, the three inner Olive
configs are wrapped behind one `AitkPython` pass
(`qwen_trtrtx_workflow.py`). The script runs each inner config, then patches
`genai_config.json` / `processor_config.json` and the tokenizer for the GenAI
runtime.

## Optimization

| Sub-model | Precision |
|-----------|-----------|
| Vision encoder | FP16 |
| Text embedding | FP16 |
| Text decoder | INT4 (block size 128, accuracy level 4) |

## Inference

Run the provided `inference_sample.ipynb`. It loads the optimized model from
`./model`, registers the NVIDIA TRT for RTX execution provider, and streams a
response for a text (and optional image) prompt.

> Metrics (latency / accuracy on a specific device) to be added after a
> benchmark run on target RTX hardware.
