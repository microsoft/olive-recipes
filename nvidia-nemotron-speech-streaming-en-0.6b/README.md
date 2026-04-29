# nvidia/nemotron-speech-streaming-en-0.6b

Olive recipes for [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b),
a 0.6B-parameter streaming English ASR model.

## Performance

- **Accuracy:** average unweighted **WER of 8.20%** across the 8 standard ESB
  [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
  benchmarks.
- **Speed:** above **2 RTFx** running on **4 cores of an Intel(R) Core(TM) Ultra 7 165H**
  (INT4 CPU build produced by this recipe).

## Recipes

- [`cpu/`](./cpu) — ONNX export + INT4 quantization for CPU execution providers.

See the README inside each subfolder for setup and run instructions.

## Future Acceptance Guidelines

When evaluating new ASR models for inclusion, the following targets apply on
the 8 standard ESB / Open ASR Leaderboard benchmarks (average unweighted WER):

- **Batch models:** WER must be **< 8%**.
- **Streaming models:** *streaming WER* (computed by concatenating the per-utterance
  output tokens emitted during streaming inference) must be **< 9%**.

A relaxation of **+1% WER** is acceptable if, on the same hardware with the same
number of CPU cores, the candidate model achieves **at least 2× the RTFx** of the
current best supported model in its category:

- Batch baseline: Whisper of similar parameter size.
- Streaming baseline: Nemotron Speech Streaming.

We also prefer models with lower long-form WER. Required long-form thresholds:

- **Earnings21:** WER must be **< 15%**.
- **TED-LIUM (long-form):** WER must be **< 5%**.

We also prefer **quantized models that fit under 2 GB on disk** — and ideally
under **1 GB**. We acknowledge this may not always be achievable for very
strong models, in which case the accuracy/RTFx criteria above take precedence.

Future models we plan to compare against (not yet integrated, but in progress):

- **Cohere Transcribe** (batch)
- **Moonshine Streaming v2** (streaming)
