# Qwen3.5-0.8B NVFP4 with NVIDIA ModelOpt

This recipe quantizes the **text-only** component of
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) using Olive's
dedicated PyTorch `ModelOptQuantization` pass. The output is a **packed Hugging Face checkpoint**,
not an ONNX model or an ONNX Runtime GenAI package. The vision encoder is not loaded
by the `text-generation` task.

## Hugging Face quantization workflow

The recipe loads the Hugging Face PyTorch model, selects ModelOpt's native `nvfp4`
preset through `quantization_config`, calibrates it, and exports with
`modelopt.torch.export.export_hf_checkpoint` (`export_format: "hf"`).
It does not use `ModelBuilder` or modify the existing ONNX quantization pass.

The pass uses `NVFP4_DEFAULT_CFG`: W4A4 E2M1 quantization with FP8 block scales,
block size 16, and ModelOpt's default calibration algorithm and module exclusions.
Embeddings, the language-model head, and other excluded modules remain in higher
precision. The KV cache is not quantized. The calibration data is 32 joined
WikiText training sequences of 512 tokens, with batch size 1.

The CUDA execution provider in `info.yml` and the workflow identifies the Olive
optimization host. Quantization runs in PyTorch; the inference script uses vLLM,
not ONNX Runtime's CUDA execution provider.

## Requirements and setup

- Linux and a CUDA-enabled PyTorch build compatible with the local NVIDIA driver.
- A CUDA GPU for calibration. **Native NVFP4 inference requires a Blackwell GPU**
  and a runtime build supporting both Qwen3.5 hybrid attention and ModelOpt NVFP4
  checkpoints. An older GPU's FP16 support alone is not sufficient.
- Transformers 5.x with Qwen3.5 support and NVIDIA ModelOpt 0.48.0 or later.
- Network access to download the model/tokenizer and WikiText dataset. Calibration
  loads the uncompressed model, so allow memory for BF16 weights and activations.
- The Olive NVFP4 implementation from
  [microsoft/Olive#2786](https://github.com/microsoft/Olive/pull/2786) on the following
  source branch (install this until the change is merged and released):

```bash
python -m pip install \
  "olive-ai @ git+https://github.com/devang-ml/Olive.git@feat/nvmodelopt-nvfp4" \
  "nvidia-modelopt[torch]>=0.48.0" "transformers>=5.3.0" datasets accelerate
```

Install the CUDA-enabled PyTorch build and ONNX Runtime GPU package appropriate
for your platform. Avoid mixing multiple ONNX Runtime packages in one environment.

For inference, use a current [vLLM](https://docs.vllm.ai/en/latest/) build supporting
Qwen3.5 and ModelOpt NVFP4. A separate environment is recommended if its PyTorch/CUDA
requirements differ from the calibration environment:

```bash
python -m pip install vllm
```

See ModelOpt's [Hugging Face PTQ guide](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq)
and [vLLM's ModelOpt integration](https://docs.vllm.ai/en/latest/features/quantization/modelopt/)
for current checkpoint and hardware compatibility.

## Quantize

Run from this directory in the calibration environment:

```bash
olive run --config Qwen3.5-0.8B_modelopt_nvfp4.json
```

Olive saves its final model descriptor to `models/nvfp4/model_config.json` and the
checkpoint to `models/nvfp4/model/`. The checkpoint contains packed safetensors,
quantization metadata, model configuration, and tokenizer files; use the checkpoint
directory, not the Olive model descriptor, for inference.

## Inference

From this directory, in the inference environment:

```bash
python inference.py \
  --model-path models/nvfp4/model \
  --prompt "What is the capital of France?" \
  --max-new-tokens 64
```

The script applies the checkpoint's chat template through `vllm.LLM.chat` and prints
the generated response. vLLM reads the quantization format from checkpoint metadata.
Generation is text-only; image/video inputs are not supported by this recipe.

## Other ModelOpt precisions and restorable checkpoints

The new pass accepts every model preset in the installed ModelOpt version, not
only NVFP4. Change `quantization_config` to a native preset such as `fp8`,
`int8_smoothquant`, `int4_awq`, `mxfp4`, `mxfp6`, `mxfp8`, or `mxint8`; a complete
native ModelOpt quantization config dictionary is also accepted. Use a distinct
`output_dir` for each variant and verify its hardware/runtime requirements.

Packed HF export is not implemented for every ModelOpt quantizer. To preserve
any native precision for further PyTorch use, set `export_format` to `modelopt`.
That saves high-precision backing weights plus quantizer state in a restorable
HF checkpoint, not a compressed deployment checkpoint. The vLLM inference script
above is intended only for compatible **packed** output. Restore ModelOpt-format
output in Python instead:

```python
import modelopt.torch.opt as mto
from transformers import AutoModelForCausalLM

mto.enable_huggingface_checkpointing()
model = AutoModelForCausalLM.from_pretrained("models/nvfp4/model")
```

## Validation status

This is an experimental recipe. Configuration validation, mocked inference-script
checks, and Olive unit tests use no model-weight downloads or GPU execution. A
native ModelOpt smoke test uses a tiny, randomly initialized Qwen3.5 model on CPU.
Additional CPU tests save and restore quantizer settings and weights for FP8,
INT8 SmoothQuant, INT4 AWQ, NVFP4, MXFP4, MXFP6, MXFP8, and MXINT8 using tiny Llama
models; CPU inference is checked only for FP8/INT8/INT4. These checks do not validate
the pretrained model's accuracy or GPU kernels. Full Qwen3.5
NVFP4 calibration, packed checkpoint loading, generation quality, and native
Blackwell inference have **not** been validated. ModelOpt's support matrix does not
currently list Qwen3.5 NVFP4 as validated; do not interpret this recipe as an
upstream compatibility or accuracy guarantee.

Small models can lose substantial accuracy at four bits. Evaluate representative
prompts against the BF16 model before deployment. Failed export or runtime support
must be resolved explicitly; this recipe does not fall back to INT4 or BF16.
