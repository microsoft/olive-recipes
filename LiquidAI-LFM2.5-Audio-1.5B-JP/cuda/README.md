# LiquidAI-LFM2.5-Audio-1.5B-JP — CUDA optimization

LFM2.5-Audio listens and speaks. ONNX Runtime GenAI runs it as the LFM2 decoder plus five graphs:

| graph | what it does |
| --- | --- |
| `model.onnx` | LFM2 decoder: `inputs_embeds` in, logits and the hidden states the audio head reads out |
| `audio_encoder.onnx` | FastConformer speech encoder and its adapter: log-mel frames to decoder features |
| `embedding.onnx` | the token table; splices the encoder features into the prompt's embeddings |
| `depthformer.onnx` | turns each decoder hidden state into one frame of 8 audio codes (80 ms of speech) |
| `audio_embedding.onnx` | feeds each frame back to the decoder as its next input |
| `audio_detokenizer.onnx` | audio codes to STFT features; `inference.py` turns them into a 24 kHz waveform |

The decoder is built by the ONNX Runtime GenAI model builder. The other graphs are built from the
checkpoint by LiquidAI's [onnx-export](https://github.com/Liquid4All/onnx-export)
(`export_audio.py`) and quantized by Olive. Every recipe writes into `model/`.

## Recipes

The two precisions follow LiquidAI's own GGUF release,
[LFM2.5-Audio-1.5B-JP-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF), which ships
Q4_0 and Q8_0 for every part of the model: symmetric round-to-nearest weights in blocks of 32, one
scale per block. The `int4` recipes are the Q4_0 equivalent and the `int8` recipes the Q8_0
equivalent.

### Decoder

#### `_cuda_int4.json` — Q4_0 equivalent
INT4 weights (symmetric, block size 32) from ONNX Runtime's default block quantizer, which rounds
exactly as Q4_0 does: the decoder's 80 INT4 matrices hold the same values as LiquidAI's Q4_0 file.
(`algo_config: rtn` rounds differently and leaves 10% more weight error.) The LM head is INT8,
standing in for the Q6_K that llama.cpp gives the tied token embedding in a Q4_0 file.

```
olive run --config LiquidAI-LFM2.5-Audio-1.5B-JP_cuda_int4.json
```

#### `_cuda_int8.json` — Q8_0 equivalent
INT8 weights (symmetric, block size 32) throughout, including the LM head. This recipe leaves
`algo_config` unset on purpose: with `precision: int8`, `algo_config: rtn` makes the model builder
emit 4-bit weights.

```
olive run --config LiquidAI-LFM2.5-Audio-1.5B-JP_cuda_int8.json
```

Both set `exclude_embeds` (the token table lives in `embedding.onnx`, so the encoder features can be
spliced in), `include_hidden_states` (the depthformer reads them) and `accuracy_level: 0`:
activations stay in floating point, as in llama.cpp's GPU kernels.

### Speech graphs

Export them once, in FP32. The export is the same for every recipe and execution provider:

```
python ../export_audio.py
```

#### `_cuda_audio_int4.json` — Q4_0 equivalent
INT4 for the encoder, the depthformer, the audio embedding and the detokenizer; the text embedding
table is INT8, again for the GGUF's Q6_K.

```
olive run --config LiquidAI-LFM2.5-Audio-1.5B-JP_cuda_audio_int4.json
```

#### `_cuda_audio_int8.json` — Q8_0 equivalent
INT8 throughout, with `accuracy_level` unset: the CUDA kernels compute in floating point.

```
olive run --config LiquidAI-LFM2.5-Audio-1.5B-JP_cuda_audio_int8.json
```

`export_audio.py` rewrites the encoder's 1x1 convolutions as `MatMul` (the output is unchanged), so
Olive quantizes them as Q4_0 and Q8_0 do; its block quantizer rewrites only `MatMul` and `Gather`.
The encoder's depthwise and subsampling convolutions stay FP32, as in the GGUF.

One thing stays FP32 where the GGUF quantizes it: **the depthformer's per-codebook tables** (code
embeddings and output heads, 34M weights, with their norms). The depthformer gathers a whole 2049 x
1024 table for each of the 8 codes of a frame, and ONNX Runtime's CPU `GatherBlockQuantized`
dequantizes it element by element: quantized, the tables made a frame 4 to 11 times slower (INT4:
485 against 42 ms on a Xeon Platinum 8358, 93 against 22 ms on an Apple M3 Ultra), and kept fewer
codes, not more.

### What each recipe quantizes

Every quantized weight is symmetric round-to-nearest in blocks of 32 with one scale per block. The
INT4 blocks are bit-identical to llama.cpp's Q4_0 on the same weights; the INT8 blocks use the full
-128..127 range where Q8_0 uses -127..127, so the values differ slightly but the error does not
(0.55% of the weights' norm for both).

| part | weights | `int4` recipe | `int8` recipe | stays FP32 | GGUF Q4_0 / Q8_0 |
| --- | --- | --- | --- | --- | --- |
| decoder projections | 1036M | INT4 | INT8 | norms, short convolutions | Q4_0 / Q8_0 |
| LM head | 134M | INT8 | INT8 | | Q6_K / Q8_0 (tied to the token table) |
| text embedding table | 134M | INT8 | INT8 | | Q6_K / Q8_0 (the same tied table) |
| speech encoder | 114M | INT4 | INT8 | depthwise and subsampling convolutions, norms | Q4_0 / Q8_0, the same convolutions F32 |
| depthformer | 68M | INT4 | INT8 | per-codebook tables (34M), norms | Q4_0 / Q8_0, tables included |
| audio embedding | 34M | INT4 | INT8 | | Q4_0 / Q8_0 |
| detokenizer | 45M | INT4 | INT8 | short convolutions, norms | Q4_0 / Q8_0 |

The LM head and the token table are one tied tensor in the GGUF; `exclude_embeds` needs them apart
here, so the package carries both.

### Assemble the package

Adds the `embedding`, `speech` and `audio_output` sections to `genai_config.json`, and gives the
embedding model the output type of the decoder's `inputs_embeds`. Run it after a decoder recipe and
an audio recipe; run it again whenever a decoder recipe rewrites `genai_config.json`.

```
python ../finalize.py
```

## Run

`inference.py` runs every mode of the model and writes speech as a 24 kHz wav:

```
python ../inference.py --mode asr --audio question.wav
python ../inference.py --mode chat --audio question.wav
python ../inference.py --mode tts --text "今日はとても良い天気ですね。" --output speech.wav
python ../inference.py --mode interleaved --audio question.wav --output answer.wav
```

The default system prompts are the model card's: `Perform ASR in japanese.`, `Perform TTS in
japanese.` and `Respond with interleaved text and audio.`; `--system` replaces them.

## Running on CUDA

Every graph runs on CUDA, the speech encoder included; its convolutions need cuDNN. The decoder
takes float16 `inputs_embeds`, and the runtime writes the embedding model's output straight into
that buffer, so `finalize.py` casts the embedding model's output to float16.

## Measured quality and speed

Measured on an NVIDIA A10. ASR is character error rate on 50 random FLEURS ja_jp test clips (10.3
minutes). TTS is 10 Japanese sentences spoken with the model card's sampling, transcribed by Whisper
large-v3-turbo and scored against the text. Punctuation and spacing are ignored.

| recipe | size | ASR CER | TTS CER |
| --- | --- | --- | --- |
| int4 | 1.22 GB | 6.94% | 2.45% |
| int8 | 1.88 GB | 7.11% | 3.07% |
| GGUF Q4_0 | 1.07 GB | 6.89% | 2.45% |
| GGUF Q8_0 | 1.82 GB | 6.94% | 1.84% |

The TTS set is 163 characters, so one character is 0.6%. The model shares its architecture and
recipes with LFM2.5-Audio-1.5B; see that model's README for speed and component fidelity.

## Setup

Python 3.12+ is required (onnx-export needs it).

```
pip install git+https://github.com/microsoft/Olive.git@refs/pull/2691/head
pip install -r requirements.txt
pip install --no-deps "liquidonnx @ git+https://github.com/Liquid4All/onnx-export.git@5a50807e6bf7bf4b025468974253328a36f69f0d"
```

onnx-export is installed without its dependencies: it depends on the CPU `onnxruntime` and
`onnxruntime-genai` wheels, which would replace the ones these recipes need. What `export_audio.py`
uses of it comes with `requirements.txt`.

These recipes need:

- Olive with [microsoft/Olive#2691](https://github.com/microsoft/Olive/pull/2691): the checkpoint's
  `config.json` has no `model_type`, and without that change Olive's `HfModel` refuses it. Until it
  is merged, install Olive from the pull request as above; afterwards from `main`.
- `onnxruntime-genai-cuda` with LFM2-Audio support,
  [microsoft/onnxruntime-genai#2601](https://github.com/microsoft/onnxruntime-genai/pull/2601), in
  both the model builder and the runtime. It is merged but not yet released; until it is, build the
  wheel from `main`.
- The CUDA build of ONNX Runtime and cuDNN (the encoder's convolutions need it). The
  `onnxruntime-gpu` 1.30 wheel on PyPI is built for CUDA 13 and needs an R580+ driver; on an older
  driver, take the CUDA 12 build from ONNX Runtime's `onnxruntime-cuda-12` feed.
