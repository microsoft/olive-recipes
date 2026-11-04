"""Run a package built by these recipes in each of the model's modes.

    python inference.py -m model --mode asr --audio question.wav
    python inference.py -m model --mode chat --audio question.wav
    python inference.py -m model --mode tts --text "Hello there." --output hello.wav
    python inference.py -m model --mode interleaved --audio question.wav --output answer.wav

ONNX Runtime GenAI generates the text and the audio codes; the audio detokenizer and an inverse
STFT, run here, turn the codes into a 24 kHz waveform.
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og

SYSTEM_PROMPTS = {
    "asr": "Perform ASR in japanese.",
    "chat": None,
    "tts": "Perform TTS in japanese.",
    "interleaved": "Respond with interleaved text and audio.",
}
# (temperature, top_k) of the audio codes, from the model card; the text is always greedy.
AUDIO_SAMPLING = {"tts": (0.8, 64), "interleaved": (1.0, 4)}

N_FFT, HOP, SAMPLE_RATE = 1280, 320, 24000


def register_webgpu_plugin(config: dict):
    """The onnxruntime wheels ship without WebGPU; the onnxruntime-ep-webgpu package provides it."""
    provider_options = config["decoder"]["session_options"]["provider_options"]
    if not any(key.lower() == "webgpu" for option in provider_options for key in option):
        return
    try:
        import onnxruntime_ep_webgpu
    except ImportError:
        return  # an onnxruntime build with WebGPU built in
    og.register_execution_provider_library(onnxruntime_ep_webgpu.get_ep_name(), onnxruntime_ep_webgpu.get_library_path())


def make_prompt(system: str | None, user: str) -> str:
    prompt = "<|startoftext|>"
    if system:
        prompt += f"<|im_start|>system\n{system}<|im_end|>\n"
    return prompt + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def to_waveform(codes: np.ndarray, detokenizer_path: Path) -> np.ndarray:
    """[num_frames, num_codebooks] audio codes -> float32 samples at SAMPLE_RATE."""
    session = ort.InferenceSession(str(detokenizer_path), providers=["CPUExecutionProvider"])
    # Only the first codebook carries the end-of-audio code; the detokenizer takes 0..2047.
    features = session.run(None, {"audio_codes": np.minimum(codes, 2047).T[None]})[0][0]
    bins = N_FFT // 2 + 1
    spectrum = (np.exp(features[:, :bins]) * np.exp(1j * features[:, bins:])).T

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
    frames = np.fft.irfft(spectrum, N_FFT, axis=0) * window[:, None]
    samples = np.zeros((frames.shape[1] - 1) * HOP + N_FFT)
    envelope = np.zeros_like(samples)
    for t in range(frames.shape[1]):
        samples[t * HOP : t * HOP + N_FFT] += frames[:, t]
        envelope[t * HOP : t * HOP + N_FFT] += window**2
    pad = (N_FFT - HOP) // 2
    return (samples[pad:-pad] / np.maximum(envelope[pad:-pad], 1e-11)).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", default="model", help="package directory (default: model)")
    parser.add_argument("--mode", choices=SYSTEM_PROMPTS, default="chat")
    parser.add_argument("--audio", help="input clip (asr, chat, interleaved)")
    parser.add_argument("--text", default="", help="input text (tts, chat, interleaved)")
    parser.add_argument("--system", help="system prompt instead of the mode's default")
    parser.add_argument("--output", default="output.wav", help="where speech is written (tts, interleaved)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="text tokens plus audio frames")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model)
    config = json.loads((model_dir / "genai_config.json").read_text())["model"]
    audio_output = config["audio_output"]
    register_webgpu_plugin(config)
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()

    system = args.system if args.system is not None else SYSTEM_PROMPTS[args.mode]
    user = ("<|audio|>" if args.audio else "") + args.text
    audios = og.Audios.open(args.audio) if args.audio else None
    inputs = processor(make_prompt(system, user), audios=audios)
    prompt_length = inputs["input_ids"].as_numpy().shape[1]

    params = og.GeneratorParams(model)
    temperature, top_k = AUDIO_SAMPLING.get(args.mode, (1.0, 4))
    params.set_search_options(
        do_sample=False,
        max_length=prompt_length + args.max_new_tokens,
        audio_interleaved=args.mode == "interleaved",
        audio_temperature=temperature,
        audio_top_k=top_k,
        random_seed=args.seed,
    )
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    while not generator.is_done():
        generator.generate_next_token()

    # Audio frames hold audio_token_id in the sequence; the switch tokens are not text either.
    skip = {config["audio_token_id"], audio_output["audio_start_token_id"], audio_output["text_end_token_id"]}
    answer = [token for token in generator.get_sequence(0)[prompt_length:] if token not in skip]
    print(tokenizer.decode(np.array(answer, np.int32)))

    codes = generator.get_output("audio_codes")
    if codes.size:
        write_wav(Path(args.output), to_waveform(codes, model_dir / "audio_detokenizer.onnx"))
        print(f"Wrote {len(codes)} audio frames ({len(codes) * HOP * 6 / SAMPLE_RATE:.1f} s) to {args.output}")


if __name__ == "__main__":
    main()
