# Whisper on WinML — Speech-to-Text on the Edge with ONNX Runtime Vitis EP

---

## Installation

### 1. Create and activate conda environment

```bash
conda create -n winml_whisper python==3.11
conda activate winml_whisper
pip install --pre -r requirements.txt
```

### 2. Check installed WASDK version

```bash
conda list | findstr wasdk
```

> **Expected Output:**
> ```
> wasdk-microsoft-windows-ai-machinelearning              1.8.260209005   pypi_0   pypi
> wasdk-microsoft-windows-applicationmodel-dynamicdependency-bootstrap   1.8.260209005   pypi_0   pypi
> ```

### 3. Install Windows App SDK

Download and install the Windows App SDK matching your `wasdk` version (e.g., `1.8.260209005`):

```bash
curl -L -o windowsappruntimeinstall-x86.exe "https://aka.ms/windowsappsdk/1.8/1.8.260209005/windowsappruntimeinstall-x86.exe"
windowsappruntimeinstall-x86.exe --quiet
```

### 4. Download ONNX encoder and Vitis AI cache (`.rai`)

Download the encoder ONNX model and the associated `.rai` cache from the Hugging Face repo that matches your model size:

| Model | Hugging Face repo |
|-------|-------------------|
| **Medium** | [amd/whisper-medium-onnx-npu](https://huggingface.co/amd/whisper-medium-onnx-npu/tree/main) — get `medium_encoder.onnx` and `ggml-medium-encoder-vitisai.rai` |

This recipe uses the pre-exported encoder ONNX from Hugging Face; it does not run Olive conversion as part of the recipe.

Place the encoder ONNX file(s) in your working directory (or a path you will pass to `--enc_onnx`).

### 5. Set up encoder cache

1. Rename `medium_encoder.onnx` to `encoder_model.onnx`.
2. Create a folder named `cacheDir` in your working directory (this path is fixed).
3. Inside `cacheDir`, create a subfolder named `encoder_model` (so you have `cacheDir/encoder_model/`).
4. Put the `.rai` file from step 4 inside `cacheDir/encoder_model/`.
5. Rename that file to `encoder_model.rai`.

---

## Run Inference and Transcribe

Use the encoder ONNX from the Hugging Face repo listed above for this recipe — set `--enc_onnx` to that model’s `encoder_model.onnx` (or your local path to it) and `--model` to the matching Whisper model name.

```bash
python run_whisper.py \
  --audio ".\audio_sample.wav" \
  --model medium \
  --enc_onnx ".\encoder_model.onnx"
```

---
## Notes
In this experiment, we are using encoder portion of the model with ORT and decoder portion of the model with Pytorch+kv cache.

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--audio` | Yes | — | Path to input audio WAV file |
| `--model` | No | `medium` | Whisper model name: `small`, `medium`, `turbo` (should match encoder ONNX) |
| `--enc_onnx` | No | `encoder_model.onnx` | Path to encoder ONNX model file |
| `--download_root` | No | Script directory | Directory to download/cache the Whisper PyTorch model |

---

## Credits

This project builds on [OpenAI Whisper](https://github.com/openai/whisper). We have borrowed code from that repository and extended it with ONNX Runtime Vitis AI EP integration, WinML execution providers, and the changes documented in this README. Whisper is licensed under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE).
