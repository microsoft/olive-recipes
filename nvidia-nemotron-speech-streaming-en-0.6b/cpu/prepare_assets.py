#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Prepare non-ONNX assets for the Nemotron Speech Streaming recipe.

This script handles steps that are outside Olive's pass pipeline:
  1. Export tokenizer (vocab.txt, tokenizer.json, tokenizer_config.json)
  2. Generate genai_config.json from the loaded NeMo model
  3. Generate audio_processor_config.json
  4. Download Silero VAD ONNX model
  5. Assemble all artifacts into the final output directory

Usage:
    python prepare_assets.py --output_dir build/output
    python prepare_assets.py --output_dir build/output --encoder_dir build/encoder --decoder_dir build/decoder --joint_dir build/joint
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPTS_DIR = SCRIPT_DIR.parent / "scripts"
MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
CHUNK_SIZE = 0.56


def export_tokenizer(model_name: str, output_dir: Path):
    """Export tokenizer using the existing export_tokenizer.py script."""
    import subprocess

    script = SCRIPTS_DIR / "export_tokenizer.py"
    cmd = [sys.executable, str(script), "--model_name", model_name, "--output_dir", str(output_dir)]
    subprocess.run(cmd, check=True)


def generate_genai_config(output_dir: Path, model_name: str = MODEL_NAME, chunk_size: float = CHUNK_SIZE):
    """Generate genai_config.json from the NeMo model."""
    import nemo.collections.asr as nemo_asr

    print("Loading NeMo model for config generation...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    asr_model.cpu().eval()

    encoder = asr_model.encoder
    decoder = asr_model.decoder
    joint = asr_model.joint

    encoder_hidden = getattr(encoder, "d_model", 1024)
    encoder_layers = getattr(encoder, "num_layers", 24)
    decoder_hidden = getattr(decoder, "pred_hidden", getattr(decoder, "d_model", 640))
    decoder_layers = getattr(decoder, "pred_rnn_layers", getattr(decoder, "num_layers", 2))
    vocab_size = joint.num_classes_with_blank
    blank_id = vocab_size - 1

    preprocessor_cfg = asr_model.cfg.get("preprocessor", {})
    sample_rate = preprocessor_cfg.get("sample_rate", 16000)
    n_mels = preprocessor_cfg.get("features", preprocessor_cfg.get("nfilt", 128))
    n_fft = preprocessor_cfg.get("n_fft", 512)
    hop_length = preprocessor_cfg.get("hop_length", 160)
    win_length = preprocessor_cfg.get("win_length", 400)
    preemph = preprocessor_cfg.get("preemph", 0.97)

    subsampling_factor = getattr(encoder, "subsampling_factor", 8)
    att_context_size = getattr(encoder, "att_context_size", None)
    left_context = att_context_size[0] if att_context_size else 70

    conv_context = 8
    if hasattr(encoder, "layers") and len(encoder.layers) > 0:
        layer = encoder.layers[0]
        if hasattr(layer, "conv") and hasattr(layer.conv, "conv"):
            conv = layer.conv.conv
            if hasattr(conv, "kernel_size"):
                ks = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
                conv_context = ks - 1

    pre_encode_cache_size = getattr(encoder, "pre_encode_cache_size", 9)
    if isinstance(pre_encode_cache_size, (list, tuple)):
        pre_encode_cache_size = pre_encode_cache_size[-1]

    chunk_samples = int(chunk_size * sample_rate)
    max_symbols = asr_model.cfg.get("decoding", {}).get("greedy", {}).get("max_symbols", 10)

    encoder_config = {
        "filename": "encoder.onnx",
        "hidden_size": encoder_hidden,
        "num_hidden_layers": encoder_layers,
        "inputs": {
            "audio_features": "audio_signal",
            "input_lengths": "length",
            "cache_last_channel": "cache_last_channel",
            "cache_last_time": "cache_last_time",
            "cache_last_channel_len": "cache_last_channel_len",
        },
        "outputs": {
            "encoder_outputs": "outputs",
            "output_lengths": "encoded_lengths",
            "cache_last_channel_next": "cache_last_channel_next",
            "cache_last_time_next": "cache_last_time_next",
            "cache_last_channel_len_next": "cache_last_channel_len_next",
        },
    }

    decoder_config = {
        "filename": "decoder.onnx",
        "hidden_size": decoder_hidden,
        "num_hidden_layers": decoder_layers,
        "inputs": {"targets": "targets", "lstm_hidden_state": "h_in", "lstm_cell_state": "c_in"},
        "outputs": {"outputs": "decoder_output", "lstm_hidden_state": "h_out", "lstm_cell_state": "c_out"},
    }

    joiner_config = {
        "filename": "joint.onnx",
        "inputs": {"encoder_outputs": "encoder_output", "decoder_outputs": "decoder_output"},
        "outputs": {"logits": "joint_output"},
    }

    vad_config = {
        "filename": "silero_vad.onnx",
        "threshold": 0.3,
        "silence_duration_ms": 3360,
        "prefix_padding_ms": 560,
    }

    config = {
        "model": {
            "type": "nemotron_speech",
            "vocab_size": vocab_size,
            "num_mels": n_mels,
            "fft_size": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "preemph": preemph,
            "log_eps": 5.96046448e-08,
            "subsampling_factor": subsampling_factor,
            "left_context": left_context,
            "conv_context": conv_context,
            "pre_encode_cache_size": pre_encode_cache_size,
            "sample_rate": sample_rate,
            "chunk_samples": chunk_samples,
            "blank_id": blank_id,
            "max_symbols_per_step": max_symbols,
            "encoder": encoder_config,
            "decoder": decoder_config,
            "joiner": joiner_config,
            "vad": vad_config,
        },
    }

    config_path = output_dir / "genai_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  [OK] Generated {config_path}")

    # Also generate audio_processor_config.json
    window_size = preprocessor_cfg.get("window_size", preprocessor_cfg.get("n_window_size", 0.025))
    window_stride = preprocessor_cfg.get("window_stride", preprocessor_cfg.get("n_window_stride", 0.01))

    if isinstance(window_size, float) and window_size < 1.0:
        window_length = int(window_size * sample_rate)
    else:
        window_length = int(window_size) if isinstance(window_size, (int, float)) else 400
    if isinstance(window_stride, float) and window_stride < 1.0:
        hop_len = int(window_stride * sample_rate)
    else:
        hop_len = int(window_stride) if isinstance(window_stride, (int, float)) else 160

    audio_config = {
        "model_type": "speech_features",
        "audio_params": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_len,
            "n_mels": n_mels,
            "window_length": window_length,
            "window_type": "hann",
            "fmin": 0,
            "fmax": sample_rate // 2,
            "dither": preprocessor_cfg.get("dither", 0.0),
            "preemphasis": preprocessor_cfg.get("preemph", 0.97),
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 1e-10,
            "normalize": preprocessor_cfg.get("normalize", "none"),
            "center": True,
            "mag_power": 2.0,
        },
    }

    audio_config_path = output_dir / "audio_processor_config.json"
    with open(audio_config_path, "w") as f:
        json.dump(audio_config, f, indent=2)
    print(f"  [OK] Generated {audio_config_path}")


def download_silero_vad(output_dir: Path):
    """Download the Silero VAD ONNX model."""
    from huggingface_hub import hf_hub_download

    print("Downloading Silero VAD model...")
    cached = hf_hub_download(repo_id="onnx-community/silero-vad", filename="onnx/model.onnx")
    dst = output_dir / "silero_vad.onnx"
    shutil.copy2(cached, str(dst))
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  [OK] Saved Silero VAD model ({size_mb:.1f} MB)")


def collect_olive_outputs(output_dir: Path, encoder_dir: Path, decoder_dir: Path, joint_dir: Path):
    """Copy Olive-produced ONNX models to the final output directory."""
    for name, src_dir in [("encoder", encoder_dir), ("decoder", decoder_dir), ("joint", joint_dir)]:
        # Olive outputs model.onnx by default
        for candidate in ["model.onnx", f"{name}.onnx"]:
            src = src_dir / candidate
            if src.exists():
                dst = output_dir / f"{name}.onnx"
                shutil.copy2(str(src), str(dst))
                # Copy external data if present
                for data_file in src_dir.glob(f"{candidate}*"):
                    if data_file != src:
                        shutil.copy2(str(data_file), str(output_dir / data_file.name.replace(candidate, f"{name}.onnx")))
                print(f"  [OK] Copied {name}.onnx from {src_dir}")
                break
        else:
            print(f"  [WARNING] {name}.onnx not found in {src_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare non-ONNX assets for Nemotron Speech recipe")
    parser.add_argument("--output_dir", type=str, default="build/output", help="Final output directory")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--encoder_dir", type=str, default="build/encoder", help="Olive encoder output dir")
    parser.add_argument("--decoder_dir", type=str, default="build/decoder", help="Olive decoder output dir")
    parser.add_argument("--joint_dir", type=str, default="build/joint", help="Olive joint output dir")
    parser.add_argument("--skip_tokenizer", action="store_true", help="Skip tokenizer export")
    parser.add_argument("--skip_configs", action="store_true", help="Skip config generation")
    parser.add_argument("--skip_vad", action="store_true", help="Skip Silero VAD download")
    parser.add_argument("--skip_collect", action="store_true", help="Skip collecting Olive outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Nemotron Speech Streaming — Asset Preparation")
    print("=" * 60)

    if not args.skip_collect:
        print("\n[1/4] Collecting Olive outputs...")
        collect_olive_outputs(
            output_dir,
            Path(args.encoder_dir),
            Path(args.decoder_dir),
            Path(args.joint_dir),
        )

    if not args.skip_tokenizer:
        print("\n[2/4] Exporting tokenizer...")
        export_tokenizer(args.model_name, output_dir)

    if not args.skip_configs:
        print("\n[3/4] Generating config files...")
        generate_genai_config(output_dir, args.model_name)

    if not args.skip_vad:
        print("\n[4/4] Downloading Silero VAD...")
        try:
            download_silero_vad(output_dir)
        except Exception as exc:
            print(f"  [WARNING] Silero VAD download failed: {exc}")
            print("  Download manually: huggingface-cli download onnx-community/silero-vad --include onnx/model.onnx")

    print(f"\n{'=' * 60}")
    print(f"Done! All artifacts in: {output_dir.resolve()}")
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
