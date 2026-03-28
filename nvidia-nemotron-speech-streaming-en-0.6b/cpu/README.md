# Nemotron Speech Streaming (CPU EP, INT4)

This recipe exports **nvidia/nemotron-speech-streaming-en-0.6b** to ONNX, optimizes the encoder, and produces CPU-ready artifacts.

## License

This model has an NVIDIA Open Model License Agreement. The contents of the license agreement can be found [here](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

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

Run all three steps as a single Olive workflow from the
`nvidia-nemotron-speech-streaming-en-0.6b` directory:

```bash
cd nvidia-nemotron-speech-streaming-en-0.6b

python -m olive run \
    --config cpu/nemotron_speech_int4_cpu_kquant.json \
    --package-config cpu/olive_package_config.json
```

`python -m olive run` must be used instead of the bare `olive run` command.
When invoked with `-m`, Python adds the current working directory to
`sys.path[0]`, which allows `cpu.olive_passes` (the custom `NemotronExport`
pass) to be imported. Running the installed `olive` script directly does not
add the CWD to `sys.path`, causing a `ModuleNotFoundError` for the pass.

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
