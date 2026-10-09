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

Run OpenVINO optimization from this `multi_comp` directory:

```bash
python optimize.py --ep ov
```

The script runs component quantization, Mobius ONNX export, and OpenVINO
optimization in sequence. QNN uses the two-machine workflow below.

`qnn.json` and `ov.json` convert the vision graph to FP16 after static
quantization, preserving its interface types. This keeps QDQ outputs consistent
when Softmax quantization introduces FP32 scales into an FP16 graph.

`gemma4_quantize.json` uses Olive's native `Gptq` pass for the decoder and
`lm_head`, a separate native `Gptq` embedding build, and `Rtn` for vision.
Decoder weights use INT4; `lm_head` is overridden to INT8. The embedding component,
including the PLE table and input projection, uses INT8. All use group size 128.
Input embeddings and the output head are untied and quantized independently.
Pass targets follow the selected components automatically; `embeds`, `lm_head`,
and `quantize_vision` do not need to be repeated in the pass configuration.
Native GPTQ requires Olive's Gemma 4 PLE, mixed-attention, and shared-KV
calibration support.

## QNN

The workflow has two user steps on two different machines:

| User step | Required hardware | Automatic work |
|---|---|---|
| 1 | CUDA GPU | `gemma4_quantize.json` -> Mobius ONNX export -> `qnn_vision_1.json` |
| 2 | Qualcomm QNN/NPU device | `qnn_vision_2.json` -> `qnn_decoder.json` -> merge into `gemma4_qnn` |

### Step 1: CUDA machine

Install the prerequisites above, `onnxruntime-gpu`, `datasets`, `fastparquet`,
`pandas`, `pillow`, and `tokenizers`. Both `gemma4_quantize.json` and vision
calibration require CUDA; QNN is not required on this machine.

```bash
python optimize.py --ep qnn --qnn-stage cuda
```

This one command:

1. Runs `olive run --config gemma4_quantize.json`.
2. Runs `olive capture-onnx-graph` to export `gemma4_onnx`.
3. Runs `qnn_vision_1.json` with `CUDAExecutionProvider`, producing
   `gemma4_qnn_vision_1`.

### Transfer

Copy the complete `gemma4_onnx` and `gemma4_qnn_vision_1` directories into
`multi_comp` on the Qualcomm device. Include every component, external-data
file, tokenizer file, and `model_config.json`, not just the vision ONNX file.
Use the same recipe scripts and configs on both machines.

The device stage rebases the first-stage package's stored paths to the local
directory, so the host and device do not need identical absolute paths.

### Step 2: QNN/NPU device

On the device, install Olive, the script dependencies, and the QNN requirements:

```bash
pip install -r requirements-qnn.txt
```

The current Python environment must provide `QNNExecutionProvider` and have
the QNN runtime/backend available for the device's HTP. CUDA is not required
on this device. Check that the `soc_model` and other provider options in
`qnn_vision_2.json` and `qnn_decoder.json` match your device.

```bash
python optimize.py --ep qnn --qnn-stage npu
```

This one command:

1. Runs `qnn_vision_2.json` on `gemma4_qnn_vision_1` to generate vision QNN
   context binaries, preserving the first-stage embedding component.
2. Runs `qnn_decoder.json` on the transferred `gemma4_onnx`, including
   export-specific node exclusions and decoder `genai_config.json` generation.
3. Combines the vision/embedding and decoder packages into `gemma4_qnn`.

The NPU step uses the current local QNN environment and does not repeat
quantization, ONNX export, or CUDA vision calibration.

### QNN output assembly

The stages use separate intermediate output directories:

- `gemma4_qnn_vision_1` is the host's calibrated vision/embedding package.
- `gemma4_qnn_vision_2` contains the compiled vision and first-stage embedding
  components plus the unmodified decoder and audio components.
- `gemma4_qnn_decoder` contains the optimized decoder plus the unmodified
  vision, embedding, and audio components copied from `gemma4_onnx`.

Two independent Olive runs cannot incrementally assemble into the same
`output_dir`. The second run detects package files written by the first and
fails because component assembly requires a clean workflow output. Olive only
automatically combines component folders when all builds belong to one
multi-build invocation. `optimize.py --ep qnn --qnn-stage npu` handles the
final merge automatically. Use clean output directories when rerunning;
the merge refuses to overwrite an existing `gemma4_qnn` package.

## Inference

Text:

```bash
python ../inference.py --model-path gemma4_qnn --prompt "What is the capital of France?" --verbose
```

Image:

```bash
python ../inference.py --model-path gemma4_qnn --image ../cat.jpeg --prompt "What animal is shown? Answer in one short sentence." --verbose
```
