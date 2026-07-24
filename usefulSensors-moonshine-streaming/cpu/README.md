# Moonshine Streaming (CPU EP, FP32)

This recipe exports **usefulsensors/moonshine-streaming-small** to ONNX and
produces CPU-ready ONNX Runtime GenAI artifacts for streaming ASR.

The streaming model is exported as **five FP32 ONNX components**, each handled
through Olive's declarative `OnnxConversion` pass (dynamo exporter, opset 20)
with the exact input/output names the streaming runner expects:

- **Frontend** – stateful convolutional feature extractor. Consumes an audio
  chunk plus rolling sample/conv buffers and emits log-mel-like features while
  updating its buffers.
- **Encoder** – sliding-window transformer encoder (features → hidden states).
- **Adapter** – positional projection (`encoded + pos_emb(arange(T) + pos_offset)`)
  producing the decoder memory.
- **Cross-KV** – precomputes per-layer cross-attention K/V from the memory once
  per segment.
- **Decoder-KV** – autoregressive decoder with cached self-attention KV and the
  precomputed cross-attention KV → logits.

## Files
- `cpu/moonshine_frontend_fp32_cpu.json` – Olive frontend config (convert only)
- `cpu/moonshine_encoder_fp32_cpu.json` – Olive encoder config (convert only)
- `cpu/moonshine_adapter_fp32_cpu.json` – Olive adapter config (convert only)
- `cpu/moonshine_cross_kv_fp32_cpu.json` – Olive cross-KV config (convert only)
- `cpu/moonshine_decoder_kv_fp32_cpu.json` – Olive decoder-KV config (convert only)
- `cpu/moonshine_encoder_kquant8_cpu.json` – Olive encoder config (convert → INT8 k-quant)
- `cpu/moonshine_decoder_kv_kquant8_cpu.json` – Olive decoder-KV config (convert → INT8 k-quant)
- `cpu/moonshine_model_load.py` – model loaders + wrapper modules + dummy inputs
- `cpu/optimize.py` – full pipeline script (Olive × 5 + tokenizer + configs + VAD)
- `cpu/export_moonshine_streaming.py` – standalone exporter (no Olive; for debugging)
- `cpu/validate_export.py` – per-component torch-vs-ONNX numeric check

## Setup
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r usefulSensors-moonshine-streaming/cpu/requirements.txt
```

## Run

From the `usefulSensors-moonshine-streaming` directory:

```bash
cd usefulSensors-moonshine-streaming

python cpu/optimize.py --output-dir build/moonshine-small
```

This runs the full pipeline:
1. **Frontend / Encoder / Adapter / Cross-KV / Decoder-KV** — Olive: `OnnxConversion` (FP32) for each of the five components
2. **Configs** — generates `genai_config.json` + `streaming_config.json`
3. **Tokenizer** — exports `tokenizer.json` + `tokenizer_config.json`
4. **VAD** — downloads Silero VAD ONNX model

`--output-dir` is resolved relative to the `cpu/` directory unless an absolute
path is given.

Export the **tiny** variant instead:

```bash
python cpu/optimize.py --model-name usefulsensors/moonshine-streaming-tiny \
    --output-dir build/moonshine-tiny
```

### Quantization (`--quantize`)

Add `--quantize` to quantize the **encoder** and **decoder_kv** MatMuls
(frontend / adapter / cross_kv stay FP32). IO names and the runtime configs
are unchanged. `--quant-method` picks the algorithm:

- `--quant-method kquant8` (default) — INT8 **weight-only k-quant**. Swaps in
  `moonshine_encoder_kquant8_cpu.json` / `moonshine_decoder_kv_kquant8_cpu.json`,
  which chain `OnnxConversion → OnnxKQuantQuantization` (`bits=8`, `block_size=32`;
  weight matmuls become `MatMulNBits`). Weight-only, so activation×activation
  attention matmuls stay FP32.
- `--quant-method kquant8-enc` — same INT8 k-quant, **encoder only**;
  `decoder_kv` stays FP32. Use when decoder quantization degrades transcription
  quality and you can afford the larger decoder.

```bash
# INT8 k-quant on encoder + decoder_kv (default)
python cpu/optimize.py --quantize --output-dir build/moonshine-small-kquant8

# INT8 k-quant on encoder only, FP32 decoder
python cpu/optimize.py --quantize --quant-method kquant8-enc \
    --output-dir build/moonshine-small-kquant8-enc
```

On a 40s clip (CPU EP), INT8 k-quant preserves transcription quality:

| build | encoder | decoder_kv | total | RTF |
|---|---|---|---|---|
| FP32 | 168 MB | 309 MB | ~541 MB | ~7.0× |
| INT8 k-quant (`--quantize`) | ~40 MB | ~120 MB | ~230 MB | ~8– 9× |
| official `.ort` | 42 MB | 174 MB | — | ~9.4× |

Transcription is essentially identical to FP32 (only quant-noise wording
drift).

Or run individual components directly with the Olive CLI:

```bash
python -m olive run --config cpu/moonshine_frontend_fp32_cpu.json
python -m olive run --config cpu/moonshine_encoder_fp32_cpu.json
python -m olive run --config cpu/moonshine_adapter_fp32_cpu.json
python -m olive run --config cpu/moonshine_cross_kv_fp32_cpu.json
python -m olive run --config cpu/moonshine_decoder_kv_fp32_cpu.json
```

## Output
Expected artifacts in `cpu/build/moonshine-small/`:
- `frontend.onnx` (+ `frontend.onnx.data`)
- `encoder.onnx` (+ `encoder.onnx.data`)
- `adapter.onnx` (+ `adapter.onnx.data`)
- `cross_kv.onnx` (+ `cross_kv.onnx.data`)
- `decoder_kv.onnx` (+ `decoder_kv.onnx.data`)
- `genai_config.json`
- `streaming_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `silero_vad.onnx`

## Validation
`validate_export.py` checks each component two ways: exported ONNX vs the
PyTorch reference outputs (tight tolerance), and — optionally — exported ONNX vs
the shipped official graph on identical inputs (loose tolerance for the
int8-quantized cross-KV / decoder-KV graphs).

It expects the `--mine` directory to contain `<component>.onnx` plus
`refs/<component>.npz` (torch reference in/out dumped by the standalone
`export_moonshine_streaming.py`). `--official` is optional:

```bash
python cpu/validate_export.py \
    --mine /path/to/moonshine-streaming-small-mine \
    --official /path/to/moonshine-streaming-small-official
```
