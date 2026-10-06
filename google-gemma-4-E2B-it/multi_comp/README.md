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

## QNN text decoder

Install the QNN requirements in addition to the prerequisites above:

```bash
pip install -r requirements-qnn.txt
```

`qnn.json` keeps the decoder, vision, embedding, and unchanged audio component
under one `CompositeModel`. The decoder pipeline incorporates the QNN text
recipe from [#645](https://github.com/microsoft/olive-recipes/pull/645):

1. Convert INT4 decoder weights to per-channel INT8 QDQ, leaving `lm_head`
   as `MatMulNBits` on CPU.
2. Apply decoder graph surgeries, convert floating-point inputs to FP16,
   and calibrate UINT16 activations / UINT8 weights with WikiText 2.
3. Split the 35 transformer layers into seven chunks plus `lm_head`, fix the
   KV cache at 1024 tokens, and build 64-token context / single-token iterator
   graphs.
4. Compile QNN context binaries with weight sharing and compose the context
   and iterator graphs.

The input config only specifies `model_path`. Olive automatically discovers
the ONNX components under `gemma4_onnx`; no component list or names need to be
maintained in the recipe. The launcher attaches `genai_config.json` only to
the discovered decoder's `additional_files` for the split pipeline, keeping
decoder metadata out of the vision calibration build.

`optimize.py --ep qnn` reads the exported decoder without loading its external
weights and resolves the `lm_head` and logits softcap exclusions from its
actual node names. Use the launcher so the decoder metadata and exclusions are
populated from your export rather than the original text recipe's node numbers.

Decoder and vision activation calibration both use `CPUExecutionProvider`.
WikiText calibration downloads the parquet shard with `huggingface_hub` and
reads it with `fastparquet`. Cauldron image calibration fetches rows and images
through the Hugging Face dataset viewer API. Both loaders share
`user_script.py` without importing `datasets` or `pyarrow`; the same vision
loader is used by the OpenVINO recipe.

The original decoder-only recipe reported the following results on Snapdragon
X Elite (X1E80100, HTP V73), with 64 generated tokens. These are not measurements
of the full multi-component pipeline:

| Metric | Value |
|---|---|
| Decode | 1.07 tok/s |
| Prefill (TTFT) | 1.15 s |
| Model load | 4.94 s |

`lm_head` remained on CPU and accounted for approximately 96% of decode latency.

## QNN vision and embedding

The vision and embedding builds incorporate
[#650](https://github.com/microsoft/olive-recipes/pull/650) in the same
`qnn.json`, without separate NPU configs or user scripts:

1. Convert the vision encoder's INT4 weights to QDQ, simplify the graph, and
   calibrate A16/W8 with the shared Cauldron loader on CPU.
2. Fix the vision input to 2,520 patches and precompile its QNN context binary.
   Precompilation avoids losing a dynamic vision graph handle when it is loaded
   alongside the decoder's precompiled context and iterator sessions.
3. Cast only the embedding outputs `inputs_embeds` and `per_layer_inputs` to
   FP16 so they match the decoder inputs. Embedding weights and inputs stay
   unchanged, and the audio component is retained.

The three QNN builds run serially on the CPU host, with QNN used for context
compilation. The assembled `gemma4_qnn` package contains the decoder stages,
precompiled vision model, FP16-output embedding model, unchanged audio model,
and shared runtime assets. OpenVINO's build pipelines are unchanged.

## Inference

Use `gemma4_quantized_onnx` for OpenVINO and `gemma4_qnn` for QNN.

Text:

```bash
python ../inference.py --model-path <output-dir> --prompt "What is the capital of France?" --verbose
```

Image:

```bash
python ../inference.py --model-path <output-dir> --image ../cat.jpeg --prompt "What animal is shown? Answer in one short sentence." --verbose
```
