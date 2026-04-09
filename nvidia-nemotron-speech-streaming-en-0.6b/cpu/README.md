# Nemotron Speech Streaming (CPU EP, INT4)

This recipe exports **nvidia/nemotron-speech-streaming-en-0.6b** to ONNX, optimizes the encoder, and produces CPU-ready artifacts.

## License

This model has an NVIDIA Open Model License Agreement. The contents of the license agreement can be found [here](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Files
- `cpu/nemotron_speech_int4_cpu_kquant.json` – Olive workflow config (export → graph fusion of the encoder)
- `cpu/olive_passes.py` – custom `NemotronExport` Olive pass that calls the export helpers directly
- `cpu/olive_package_config.json` – registers `NemotronExport` so `olive run` can resolve it
- `cpu/optimize.py` – end-to-end Python script: export + Olive graph fusion + INT4 k-quant (via `MatMulNBitsQuantizer` with `KQuantWeightOnlyQuantConfig`) + supporting files + Silero VAD
- `scripts/export_nemotron_to_onnx_static_shape.py` – ONNX export (streaming/static-shape), callable as `export_to_onnx()`
- `scripts/export_tokenizer.py` – tokenizer export, callable as `export_tokenizer()`
- `scripts/test_e2e.py` – e2e smoke test

## Setup
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nvidia-nemotron-speech-streaming-en-0.6b/cpu/requirements.txt
```

## Run

### Option A — Olive workflow (export → graph fusion)

Run the export and Conformer graph fusion as a single Olive workflow from the
`nvidia-nemotron-speech-streaming-en-0.6b` directory.  This produces a
graph-fused `model.onnx` in `build/onnx_models_fused/`; use Option B for the
complete INT4 k-quant artifact set.

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

### Option B — Python script (recommended)

Runs the complete pipeline: NeMo export → Olive graph fusion → INT4 k-quant
quantization (via `MatMulNBitsQuantizer` with `KQuantWeightOnlyQuantConfig`
directly) → copy supporting files (decoder, joint, configs, tokenizer) →
Silero VAD download.

```bash
cd nvidia-nemotron-speech-streaming-en-0.6b

python cpu/optimize.py
```

To skip the NeMo export step if models are already in `build/onnx_models_fp32/`:

```bash
python cpu/optimize.py --skip-export
```

## Output
Expected optimized artifacts (produced by Option B / `optimize.py`) in:
- `build/onnx_models_int4/encoder.onnx`
- `build/onnx_models_int4/decoder.onnx`
- `build/onnx_models_int4/joint.onnx`
- `build/onnx_models_int4/silero_vad.onnx`
- `build/onnx_models_int4/genai_config.json`
- `build/onnx_models_int4/audio_processor_config.json`
- `build/onnx_models_int4/tokenizer.json`
- `build/onnx_models_int4/tokenizer_config.json`
- `build/onnx_models_int4/vocab.txt`
