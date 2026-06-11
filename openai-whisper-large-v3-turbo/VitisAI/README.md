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

Download the encoder ONNX model from the Hugging Face repo:

| Model | Hugging Face repo |
|-------|-------------------|
| **Large-v3-turbo** | [amd/whisper-large-turbo-onnx-npu](https://huggingface.co/amd/whisper-large-turbo-onnx-npu/tree/main) — get `encoder_model.onnx`, `encoder_model.onnx.data` (if present) |

Place the encoder ONNX file(s) in your working directory (or a path you will pass to `--enc_onnx`).

---

## Run Inference and Transcribe

Use the encoder ONNX from one of the three Hugging Face repos above (medium, small, or large-v3-turbo) depending on your task — set `--enc_onnx` to that model’s `encoder_model.onnx` (or your local path to it) and `--model` to the matching Whisper model name.

```bash
python run_whisper.py \
  --audio ".\audio_sample.wav" \
  --model turbo \
  --enc_onnx ".\encoder_model.onnx"
```
---
## Notes
In this experiment, we are using encoder portion of the model with ORT and decoder portion of the model with Pytorch+kv cache.

## Command-Line Arguments

---

| Argument | Required | Default | Description |
|---|---|---|---|
| `--audio` | Yes | — | Path to input audio WAV file |
| `--model` | No | `turbo` | Whisper model name: `small`, `medium`, `turbo` (should match encoder ONNX) |
| `--enc_onnx` | No | `encoder_model.onnx` | Path to encoder ONNX model file |

---

## Credits

This project builds on [OpenAI Whisper](https://github.com/openai/whisper). We have borrowed code from that repository and extended it with ONNX Runtime Vitis AI EP integration, WinML execution providers, and the changes documented in this README. Whisper is licensed under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE).
