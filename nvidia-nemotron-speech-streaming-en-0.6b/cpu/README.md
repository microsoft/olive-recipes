# Nemotron Speech Streaming (CPU EP, INT4)

This recipe exports **nvidia/nemotron-speech-streaming-en-0.6b** to ONNX, optimizes the encoder, and produces CPU-ready artifacts.

## Files
- `cpu/nemotron_speech_int4_cpu_kquant.json` – Olive workflow config (export → graph fusion → INT4 quantization)
- `cpu/olive_passes.py` – custom `NemotronExport` Olive pass that wraps the export script
- `cpu/olive_package_config.json` – registers `NemotronExport` so `olive run` can resolve it
- `scripts/export_nemotron_to_onnx_static_shape.py` – ONNX export (streaming/static-shape)
- `scripts/optimize_encoder.py` – encoder graph fusion + dtype conversion/quantization
- `scripts/test_e2e.py` – e2e smoke test

## Setup
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nvidia-nemotron-speech-streaming-en-0.6b/cpu/requirements.txt
```

## Run

### Option A — full Olive workflow (export → fusion → INT4)

Run all three steps as a single `olive run` workflow from the
`nvidia-nemotron-speech-streaming-en-0.6b` directory:

```bash
cd nvidia-nemotron-speech-streaming-en-0.6b

PYTHONPATH=. olive run \
    --config cpu/nemotron_speech_int4_cpu_kquant.json \
    --package-config cpu/olive_package_config.json
```

`PYTHONPATH=.` is required so that Python can find `cpu.olive_passes`
(the custom `NemotronExport` pass) when `olive` is run as an installed
script. Without it, the module will not be on `sys.path`.

The `--package-config` flag registers the custom `NemotronExport` pass
(defined in `cpu/olive_passes.py`) with Olive so the export step can be
executed as a first-class Olive pass.

### Option B — Python script

```bash
cd nvidia-nemotron-speech-streaming-en-0.6b

python cpu/optimize.py
```

To skip the NeMo export step if models are already in `build/onnx_models_fp32/`:

```bash
python cpu/optimize.py --skip-export
```

## Output
Expected optimized artifacts in:
- `build/onnx_models_int4/encoder.onnx`
- `build/onnx_models_int4/decoder.onnx`
- `build/onnx_models_int4/joint.onnx`
- `build/onnx_models_int4/genai_config.json`
- `build/onnx_models_int4/audio_processor_config.json`
