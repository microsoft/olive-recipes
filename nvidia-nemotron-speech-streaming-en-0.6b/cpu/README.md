# Nemotron Speech Streaming (CPU, INT4 k-quant)

Optimizes **nvidia/nemotron-speech-streaming-en-0.6b** for CPU inference using
Olive's built-in passes — no custom passes or orchestration scripts required.

## License

This model has an NVIDIA Open Model License Agreement. See
[license details](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Architecture

The model is an RNNT (Recurrent Neural Network Transducer) with three components:

| Component | Description | Olive Pipeline |
|-----------|-------------|----------------|
| **Encoder** | Conformer (~600 MB), streaming with cache I/O | Export (dynamo) → Graph Fusion → INT4 k-quant |
| **Decoder** | LSTM with explicit h/c states (tiny) | Export only |
| **Joint** | Linear projection network (tiny) | Export only |

Each component has its own Olive JSON config and is optimized independently.

## Setup

```bash
cd nvidia-nemotron-speech-streaming-en-0.6b
python -m venv .venv && source .venv/bin/activate
pip install -r cpu/requirements.txt
```

## Run

### Step 1 — Export and optimize each component with Olive

```bash
# Encoder: export → graph fusion → INT4 k-quant quantization
olive run --config cpu/encoder.json

# Decoder: export to ONNX (FP32)
olive run --config cpu/decoder.json

# Joint: export to ONNX (FP32)
olive run --config cpu/joint.json
```

### Step 2 — Prepare supporting assets

Collect the Olive outputs, export the tokenizer, generate config files,
and download the Silero VAD model:

```bash
python cpu/prepare_assets.py \
    --encoder_dir build/encoder \
    --decoder_dir build/decoder \
    --joint_dir build/joint \
    --output_dir build/output
```

## Output

All artifacts in `build/output/`:

| File | Description |
|------|-------------|
| `encoder.onnx` (+`.data`) | INT4 k-quant quantized, graph-fused Conformer encoder |
| `decoder.onnx` | FP32 stateful LSTM decoder |
| `joint.onnx` | FP32 joint/joiner network |
| `silero_vad.onnx` | Silero Voice Activity Detection model |
| `genai_config.json` | ORT GenAI model configuration |
| `audio_processor_config.json` | Audio preprocessing parameters |
| `tokenizer.json` | HuggingFace Unigram tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `vocab.txt` | Token vocabulary |

## Files

| File | Purpose |
|------|---------|
| `cpu/encoder.json` | Olive config: encoder export + optimization + quantization |
| `cpu/decoder.json` | Olive config: decoder export |
| `cpu/joint.json` | Olive config: joint network export |
| `cpu/user_script.py` | Model loaders, IO configs, dummy inputs for all components |
| `cpu/prepare_assets.py` | Tokenizer export, config generation, VAD download |
| `scripts/export_tokenizer.py` | NeMo tokenizer → HuggingFace Unigram conversion |
