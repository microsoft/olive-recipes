# MIT License
#
# Copyright (C) 2026, Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (C) [2026] Advanced Micro Devices, Inc. All Rights Reserved.

import time
import os
import argparse
import json
import sys
import numpy as np
import whisper
import subprocess
import onnxruntime as ort
from scipy.io import wavfile
from scipy.signal import resample

SAMPLE_RATE = 16000  # Whisper expects 16 kHz

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def register_execution_providers(script_dir=None):
    """Register WinML execution providers. script_dir: directory containing winml.py (default: this script's dir)."""
    base = script_dir or _SCRIPT_DIR
    worker_script = os.path.join(base, "winml.py")
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    for name, lib_path in paths.items():
        if not lib_path or not os.path.exists(lib_path):
            continue
        ort.register_execution_provider_library(name, lib_path)


def load_audio_no_ffmpeg(path: str) -> np.ndarray:
    """Load WAV as mono 16 kHz float32 (no ffmpeg, no torchcodec). Uses scipy."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / sr)
        data = resample(data, n).astype(np.float32)
    return data


def run_whisper(
    audio_path: str = None,
    audio: np.ndarray = None,
    *,
    enc_onnx: str = "encoder_model.onnx",
    enc_cache_dir: str = "cacheDir",
    vitisai_config: str = "vitisai_config.json",
    model: str = "small",
    download_root: str = None,
) -> dict:
    """
    Run Whisper E2E: load audio, run encoder (ONNX on NPU) + decoder, return transcription and metrics.

    Provide either audio_path (path to WAV) or audio (float32 mono 16 kHz array). All other args are optional.

    Returns dict with: text, detected_language, rtf, elapsed_sec, audio_duration_sec.
    """
    if audio_path is None and audio is None:
        raise ValueError("Provide either audio_path or audio")
    if audio_path is not None and audio is not None:
        raise ValueError("Provide only one of audio_path or audio")

    if download_root is None:
        download_root = _SCRIPT_DIR

    if audio_path is not None:
        audio = load_audio_no_ffmpeg(audio_path)
    audio_duration_sec = min(audio.shape[0] / SAMPLE_RATE, 30.0)
    audio = whisper.pad_or_trim(audio)

    model_obj = whisper.load_model(model, download_root=download_root)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model_obj.dims.n_mels).to(model_obj.device)

    _, probs = model_obj.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    register_execution_providers()

    cache_key = "encoder_model"
    options = whisper.DecodingOptions(
        enc_use_onnx=bool(enc_onnx),
        enc_onnx_fname=enc_onnx or "",
        use_winml=True,
        enc_use_vitis=True,
        enc_cache_dir=enc_cache_dir,
        enc_cache_key=cache_key,
        enc_config_json=vitisai_config,
    )

    t0 = time.perf_counter()
    result = whisper.decode(model_obj, mel, options)
    elapsed_sec = time.perf_counter() - t0
    rtf = elapsed_sec / audio_duration_sec
    return {
        "text": result.text,
        "detected_language": detected_language,
        "rtf": rtf,
        "elapsed_sec": elapsed_sec,
        "audio_duration_sec": audio_duration_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enc_onnx",
        type=str,
        default="encoder_model.onnx",
        help="Path to encoder ONNX model file",
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio WAV file")
    parser.add_argument(
        "--download_root",
        type=str,
        default=None,
        help="Directory to download/cache Whisper PyTorch model (default: script directory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["small", "medium", "turbo"],
        help="Whisper model name for load_model (default: medium)",
    )
    args = parser.parse_args()

    out = run_whisper(
        audio_path=args.audio,
        enc_onnx=args.enc_onnx,
        model=args.model,
        download_root=args.download_root,
    )
    print("\n")
    print("Transcription results:")
    print(f"Detected language: {out['detected_language']}")
    print(f"RTF: {out['rtf']:.4f}  (decode: {out['elapsed_sec']:.3f}s, audio: {out['audio_duration_sec']:.3f}s)")
    print(out["text"])


if __name__ == "__main__":
    main()
