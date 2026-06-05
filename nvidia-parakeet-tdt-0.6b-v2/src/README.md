# Parakeet TDT 0.6B v2 (CPU + CUDA)

This recipe exports **nvidia/parakeet-tdt-0.6b-v2** to ONNX with an INT4-quantized encoder and FP32 decoder/joint. The same ONNX graphs work on both CPU and CUDA — the `--device` flag only selects which provider is written into `genai_config.json`.

## Files
- `src/parakeet_encoder_int4.json` – Olive workflow config for the encoder (OnnxConversion → graph fusion → INT4 quantization)
- `src/parakeet_decoder_fp32.json` – Olive workflow config for the decoder (FP32)
- `src/parakeet_joint_fp32.json` – Olive workflow config for the joint network (FP32)
- `src/parakeet_model_load.py` – model loader script for Olive
- `src/optimize.py` – full pipeline script (Olive pipelines + genai_config assembly)
- `src/requirements.txt` – Python dependencies

## Setup
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nvidia-parakeet-tdt-0.6b-v2/src/requirements.txt
```

## Run

From the `nvidia-parakeet-tdt-0.6b-v2` directory:

```bash
cd nvidia-parakeet-tdt-0.6b-v2

# CPU target (default)
python src/optimize.py

# CUDA target (writes CUDAExecutionProvider into genai_config.json)
python src/optimize.py --device cuda
```

Useful flags:
- `--model-name <hf-id-or-.nemo-path>` – source model (defaults to `nvidia/parakeet-tdt-0.6b-v2`)
- `--output-dir <path>` – output directory (default: `build/parakeet-tdt-0.6b-v2-onnx-int4`)
- `--skip-configs` – only export the ONNX files, skip `genai_config.json` / tokenizer / VAD assembly
- `--device cpu|cuda` – EP for the generated `genai_config.json` (ONNX graphs are identical)

The pipeline:
1. **Encoder** — convert → fuse → INT4 quantize via `parakeet_encoder_int4.json`
2. **Decoder** — convert (FP32) via `parakeet_decoder_fp32.json`
3. **Joint** — convert (FP32) via `parakeet_joint_fp32.json`
4. **Assemble** — write `genai_config.json` with session_options for the selected device and copy tokenizer / configs into the output directory
