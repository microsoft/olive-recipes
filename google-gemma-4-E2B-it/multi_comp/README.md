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

Run one command from this `multi_comp` directory:

```bash
python optimize.py --ep ov
python optimize.py --ep qnn
```

The script runs component quantization, Mobius ONNX export, and the selected
execution-provider optimization in sequence.

## QNN

Install the QNN requirements in addition to the prerequisites above:

```bash
pip install -r requirements-qnn.txt
```

Before running QNN:

- The current Python environment must provide `CUDAExecutionProvider` for
  vision calibration and be able to execute the QNN context-binary pass.
- `qnn_decoder.json` requires a Qualcomm QNN device. Set
  `systems.qnn_system.python_environment_path` to the QNN Python environment
  on that device.

Then run one command:

```bash
python optimize.py --ep qnn
```

The command automatically:

1. Quantizes and exports the shared `gemma4_onnx` model.
2. Runs `qnn_vision.json` for the vision and embedding components.
3. Runs `qnn_decoder.json` for the decoder, including export-specific node
   exclusions and decoder `genai_config.json` generation.
4. Combines both component packages into the final `gemma4_qnn` directory.

### QNN output assembly

Internally, the two Olive runs still use different intermediate output
directories:

- `gemma4_qnn_vision` contains optimized vision and embedding components plus
  the unmodified decoder and audio components copied from `gemma4_onnx`.
- `gemma4_qnn_decoder` contains the optimized decoder plus the unmodified
  vision, embedding, and audio components copied from `gemma4_onnx`.

Two independent Olive runs cannot incrementally assemble into the same
`output_dir`. The second run detects package files written by the first and
fails because component assembly requires a clean workflow output. Olive only
automatically combines component folders when all builds belong to one
multi-build invocation. `optimize.py --ep qnn` handles the separate
intermediate directories and final merge automatically.

## Inference

Text:

```bash
python ../inference.py --model-path gemma4_qnn --prompt "What is the capital of France?" --verbose
```

Image:

```bash
python ../inference.py --model-path gemma4_qnn --image ../cat.jpeg --prompt "What animal is shown? Answer in one short sentence." --verbose
```
