# Gemma 4 E2B - Multi-component Optimization

This recipe optimizes for
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it):

## Prerequisites

```bash
pip install "git+https://github.com/microsoft/Olive.git"
pip install "git+https://github.com/onnxruntime/mobius.git"
pip install transformers torch onnxruntime-genai requests
hf auth login
```

## Optimize

Run OpenVINO from this `multi_comp` directory:

```bash
python optimize.py --ep ov
```

The script runs component quantization, Mobius ONNX export, and OpenVINO
optimization in sequence. The result is written to `gemma4_ov`.

## QNN split workflow

Install the QNN requirements in addition to the prerequisites above:

```bash
pip install -r requirements-qnn.txt
```

The QNN workflow is split because its two component groups require different
machines:

- `qnn_vision.json` requires `CUDAExecutionProvider` for vision calibration.
  Run it in a CUDA environment that can also execute the QNN context-binary
  pass.
- `qnn_decoder.json` requires a Qualcomm QNN device. Set
  `systems.qnn_system.python_environment_path` to the QNN Python environment
  on that device before running it.

Run the vision and embedding build on the CUDA machine:

```bash
python optimize.py --ep qnn-vision
```

This creates the shared `gemma4_onnx` input and the
`gemma4_qnn_vision` package. Copy `gemma4_onnx` to the QNN device so both
component jobs use the same exported model, then run:

```bash
python optimize.py --ep qnn-decoder --skip-prepare
```

Without `--skip-prepare`, the decoder command regenerates the quantized model
and ONNX export before running `qnn_decoder.json`. The decoder launcher also
resolves the export-specific `lm_head` and logits-softcap node exclusions and
attaches `genai_config.json` to the decoder pipeline.

Put `gemma4_qnn_vision` and `gemma4_qnn_decoder` next to `optimize.py` on one
machine, then assemble the deployable package:

```bash
python optimize.py --merge-qnn
```

The final package is written to `gemma4_qnn`. The merge starts from the decoder
package so its generated `genai_config.json` is preserved, replaces the
`embedding` and `vision_encoder` directories with their CUDA-job outputs, keeps
the unchanged audio component, and rewrites `model_config.json` for the final
path.

### Why the QNN outputs must be different

Olive's component assembly produces a complete package for each invocation:

- `gemma4_qnn_vision` contains optimized vision and embedding components plus
  the unmodified decoder and audio components copied from `gemma4_onnx`.
- `gemma4_qnn_decoder` contains the optimized decoder plus the unmodified
  vision, embedding, and audio components copied from `gemma4_onnx`.

Two independent Olive runs cannot incrementally assemble into the same
`output_dir`. The second run detects package files written by the first and
fails because component assembly requires a clean workflow output. Olive only
automatically combines component folders when all builds belong to one
multi-build invocation. Since these builds run on different hardware, keep the
two intermediate output directories and use `--merge-qnn`.

## Inference

Text:

```bash
python ../inference.py --model-path gemma4_qnn --prompt "What is the capital of France?" --verbose
```

Image:

```bash
python ../inference.py --model-path gemma4_qnn --image ../cat.jpeg --prompt "What animal is shown? Answer in one short sentence." --verbose
```
