#!/usr/bin/env python3
"""
Test Nemotron ASR with real speech audio.
Compares raw ONNX Runtime inference with onnxruntime-genai pipeline.
"""

import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test Nemotron ASR with real speech audio.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).parent / "onnx_models"),
        help="Path to directory containing ONNX models (default: scripts/onnx_models).",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "test" / "test_models" / "audios" / "jfk.flac"),
        help="Path to the input audio file (default: test/test_models/audios/jfk.flac).",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    audio_path = Path(args.audio_path)

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return False

    # Load audio
    waveform_np, sr = sf.read(str(audio_path), dtype="float32")
    print(f"Audio: shape={waveform_np.shape}, sr={sr}, duration={len(waveform_np)/sr:.2f}s")

    if len(waveform_np.shape) > 1:
        waveform_np = waveform_np.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import torchaudio
        waveform_t = torch.from_numpy(waveform_np).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform_t = resampler(waveform_t)
        waveform_np = waveform_t.squeeze(0).numpy()
        sr = 16000
        print(f"Resampled to 16kHz, length={len(waveform_np)}")

    # Use NeMo preprocessor for mel features
    waveform_t = torch.from_numpy(waveform_np).unsqueeze(0)
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

    preprocessor = FilterbankFeatures(
        sample_rate=16000,
        n_window_size=400,   # 25ms
        n_window_stride=160, # 10ms
        n_fft=512,
        nfilt=128,
        dither=0.0,
        pad_to=0,
        normalize="per_feature",
    )
    length = torch.tensor([waveform_t.shape[1]])
    features, feat_length = preprocessor(waveform_t, length)
    mel_np = features.numpy().astype(np.float32)
    print(f"Mel features: {mel_np.shape}, feat_length={feat_length.item()}")

    # ---- Raw ONNX Runtime Greedy Decode (streaming encoder) ----
    import onnxruntime as ort

    enc = ort.InferenceSession(str(model_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(model_dir / "decoder.onnx"), providers=["CPUExecutionProvider"])
    jnt = ort.InferenceSession(str(model_dir / "joint.onnx"), providers=["CPUExecutionProvider"])

    # Streaming encoder constants (chunk_size=0.56s)
    n_layers = 24
    d_model = 1024
    mel_features = 128
    chunk_mel_frames = 56
    pre_encode_cache = 9
    static_mel_frames = chunk_mel_frames + pre_encode_cache  # 65
    last_channel_cache_size = 70  # left_chunks(10) * chunk_encoded_frames(7)
    conv_context = 8

    # mel_np is [1, 128, T] from NeMo preprocessor; transpose to [1, T, 128]
    mel_t = mel_np.transpose(0, 2, 1)  # [1, T, 128]
    total_frames = mel_t.shape[1]

    # Initialize caches
    cache_ch = np.zeros((1, n_layers, last_channel_cache_size, d_model), dtype=np.float32)
    cache_tm = np.zeros((1, n_layers, d_model, conv_context), dtype=np.float32)
    cache_len_val = np.zeros((1,), dtype=np.int64)

    # Process audio in chunks, accumulate encoded frames
    all_encoded = []
    for start in range(0, total_frames, chunk_mel_frames):
        end = min(start + chunk_mel_frames, total_frames)
        chunk = mel_t[:, start:end, :]  # [1, <=56, 128]

        # Prepend pre-encode cache (zeros for first chunk, previous frames otherwise)
        if start == 0:
            prefix = np.zeros((1, pre_encode_cache, mel_features), dtype=np.float32)
        else:
            prefix_start = max(0, start - pre_encode_cache)
            prefix = mel_t[:, prefix_start:start, :]
            if prefix.shape[1] < pre_encode_cache:
                pad = np.zeros((1, pre_encode_cache - prefix.shape[1], mel_features), dtype=np.float32)
                prefix = np.concatenate([pad, prefix], axis=1)

        # Pad chunk to full size if needed (last chunk)
        if chunk.shape[1] < chunk_mel_frames:
            pad = np.zeros((1, chunk_mel_frames - chunk.shape[1], mel_features), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=1)

        audio_input = np.concatenate([prefix, chunk], axis=1)  # [1, 65, 128]
        actual_length = min(end - start + pre_encode_cache, static_mel_frames)
        length_input = np.array([actual_length], dtype=np.int64)

        enc_out = enc.run(None, {
            "audio_signal": audio_input, "length": length_input,
            "cache_last_channel": cache_ch, "cache_last_time": cache_tm,
            "cache_last_channel_len": cache_len_val,
        })
        encoded_chunk = enc_out[0]  # [1, T', 1024]
        enc_chunk_len = int(enc_out[1][0])
        cache_ch, cache_tm, cache_len_val = enc_out[2], enc_out[3], enc_out[4]

        # Only keep valid encoded frames
        all_encoded.append(encoded_chunk[:, :enc_chunk_len, :])

    encoded_t = np.concatenate(all_encoded, axis=1)  # [1, total_enc_frames, 1024]
    enc_len = encoded_t.shape[1]
    print(f"Encoder: enc_len={enc_len}")

    # Stateful RNNT greedy decode
    hidden_size = 640
    num_lstm_layers = 2
    h = np.zeros((num_lstm_layers, 1, hidden_size), dtype=np.float32)
    c = np.zeros((num_lstm_layers, 1, hidden_size), dtype=np.float32)

    tokens_raw = []
    cur = 0
    for t in range(enc_len):
        enc_t = encoded_t[:, t : t + 1, :]
        for _ in range(10):
            tgt = np.array([[cur]], dtype=np.int64)
            d = dec.run(None, {"targets": tgt, "h_in": h, "c_in": c})
            dh = d[0].transpose(0, 2, 1)  # [1, 640, 1] -> [1, 1, 640]
            h, c = d[1], d[2]
            j = jnt.run(None, {"encoder_output": enc_t, "decoder_output": dh})
            tok = int(np.argmax(j[0].squeeze()))
            if tok == 1024:
                break
            tokens_raw.append(tok)
            cur = tok

    # ---- onnxruntime-genai ----
    import onnxruntime_genai as og

    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=512, batch_size=1)
    gen = og.Generator(model, params)

    inputs = og.NamedTensors()
    inputs["audio_signal"] = mel_np
    inputs["input_ids"] = np.array([[0]], dtype=np.int32)
    gen.set_inputs(inputs)

    step = 0
    while not gen.is_done():
        gen.generate_next_token()
        step += 1
        if step > 500:
            break

    tok_list = list(gen.get_sequence(0))
    text_ids_og = [t for t in tok_list[1:] if t != 1024]

    # Compare and decode
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Raw ONNX tokens ({len(tokens_raw)}): {tokens_raw}")
    print(f"OG tokens       ({len(text_ids_og)}): {text_ids_og}")
    print(f"Token match: {tokens_raw == text_ids_og}")

    if tokens_raw:
        text = tokenizer.decode(np.array(tokens_raw, dtype=np.int32))
        print(f'\nTranscription (raw ONNX): "{text}"')
    if text_ids_og:
        text = tokenizer.decode(np.array(text_ids_og, dtype=np.int32))
        print(f'Transcription (OG):       "{text}"')
    if not tokens_raw and not text_ids_og:
        print("Both paths: all blanks (no speech tokens)")

    return tokens_raw == text_ids_og


if __name__ == "__main__":
    success = main()
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
