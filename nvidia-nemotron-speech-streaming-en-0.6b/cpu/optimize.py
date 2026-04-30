"""End-to-end optimization pipeline for Nemotron Speech Streaming.

All model components (encoder, decoder, joint) are exported and optimized
using Olive's declarative pass system:

  - Encoder: OnnxConversion → OrtTransformersOptimization → OnnxKQuantQuantization
  - Decoder: OnnxConversion (FP32)
  - Joint:   OnnxConversion (FP32)

After the Olive pipelines, tokenizer and config files are generated and
Silero VAD is downloaded.

Usage:
    # Full pipeline
    python cpu/optimize.py

    # Or use Olive CLI directly for individual components:
    python -m olive run --config cpu/nemotron_speech_int4_cpu.json
    python -m olive run --config cpu/nemotron_decoder_fp32_cpu.json
    python -m olive run --config cpu/nemotron_joint_fp32_cpu.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure the recipe root is on sys.path so `from cpu.nemotron_model_load import ...` works
# regardless of where the script is invoked from.
_SCRIPT_DIR = Path(__file__).resolve().parent
_RECIPE_ROOT = _SCRIPT_DIR.parent
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

_SCRIPT_DIR = Path(__file__).parent.resolve()
_TOKENIZER_SCRIPT = _SCRIPT_DIR.parent / "scripts" / "export_tokenizer.py"

DEFAULT_OUTPUT_DIR = "build/onnx_models_int4"


def _resolve(path: str) -> Path:
    """Resolve a path relative to the cpu/ directory."""
    p = Path(path)
    return p if p.is_absolute() else _SCRIPT_DIR / p


def _run_olive_pipeline(config_name: str, output_dir: str, output_subdir: str):
    """Run an Olive pipeline from a JSON config, overriding output_dir."""
    from olive import run as olive_run

    config_path = _SCRIPT_DIR / config_name
    with open(config_path) as f:
        config = json.load(f)

    config["output_dir"] = str(_resolve(output_dir) / output_subdir)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(_SCRIPT_DIR), delete=False
    ) as tmp:
        json.dump(config, tmp, indent=4)
        tmp_path = tmp.name

    try:
        olive_run(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_olive_pipelines(output_dir: str):
    """Run all Olive pipelines: encoder (INT4), decoder (FP32), joint (FP32)."""
    print("=== Stage 1: Olive Encoder (OnnxConversion → fusion → INT4 quant) ===")
    _run_olive_pipeline("nemotron_speech_int4_cpu.json", output_dir, "encoder.onnx")
    print()

    print("=== Stage 2: Olive Decoder (OnnxConversion, FP32) ===")
    _run_olive_pipeline("nemotron_decoder_fp32_cpu.json", output_dir, "decoder.onnx")
    print()

    print("=== Stage 3: Olive Joint (OnnxConversion, FP32) ===")
    _run_olive_pipeline("nemotron_joint_fp32_cpu.json", output_dir, "joint.onnx")
    print()


def run_tokenizer_export(model_name: str, output_dir: str):
    """Export tokenizer files to the output directory."""
    print("=== Stage 4: Exporting tokenizer ===")
    cmd = [
        sys.executable,
        str(_TOKENIZER_SCRIPT),
        "--model_name", model_name,
        "--output_dir", str(_resolve(output_dir)),
    ]
    result = subprocess.run(cmd, cwd=str(_SCRIPT_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"Tokenizer export failed (exit code {result.returncode})")
    print()


def generate_configs(model_name: str, output_dir: str, chunk_size: float):
    """Generate genai_config.json and audio_processor_config.json.

    Loads the NeMo model to extract architecture parameters, then writes
    the config files needed by onnxruntime-genai for inference.
    """
    print("=== Stage 5: Generating config files ===")
    import nemo.collections.asr as nemo_asr
    from cpu.nemotron_model_load import get_att_context_size, D_MODEL, N_LAYERS, DECODER_HIDDEN, DECODER_LSTM_LAYERS

    if model_name.endswith(".nemo"):
        asr_model = nemo_asr.models.ASRModel.restore_from(model_name)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    asr_model.eval()

    dst = _resolve(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    encoder = asr_model.encoder
    decoder = asr_model.decoder
    joint = asr_model.joint

    vocab_size = joint.num_classes_with_blank
    blank_id = vocab_size - 1

    preprocessor_cfg = asr_model.cfg.get('preprocessor', {})
    sample_rate = preprocessor_cfg.get('sample_rate', 16000)
    n_mels = preprocessor_cfg.get('features', preprocessor_cfg.get('nfilt', 128))
    n_fft = preprocessor_cfg.get('n_fft', 512)
    hop_length = preprocessor_cfg.get('hop_length', 160)
    win_length = preprocessor_cfg.get('win_length', 400)
    preemph = preprocessor_cfg.get('preemph', 0.97)

    subsampling_factor = getattr(encoder, 'subsampling_factor', 8)
    att_context_size = get_att_context_size(chunk_size)
    left_context = att_context_size[0]

    conv_context = 8
    if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
        layer = encoder.layers[0]
        if hasattr(layer, 'conv') and hasattr(layer.conv, 'conv'):
            conv = layer.conv.conv
            if hasattr(conv, 'kernel_size'):
                ks = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
                conv_context = ks - 1

    pre_encode_cache_size = getattr(encoder, 'pre_encode_cache_size', 9)
    if isinstance(pre_encode_cache_size, (list, tuple)):
        pre_encode_cache_size = pre_encode_cache_size[-1]

    chunk_samples = int(chunk_size * sample_rate)
    max_symbols = asr_model.cfg.get('decoding', {}).get('greedy', {}).get('max_symbols', 10)

    genai_config = {
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
            "encoder": {
                "filename": "encoder.onnx",
                "hidden_size": D_MODEL,
                "num_hidden_layers": N_LAYERS,
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
            },
            "decoder": {
                "filename": "decoder.onnx",
                "hidden_size": DECODER_HIDDEN,
                "num_hidden_layers": DECODER_LSTM_LAYERS,
                "inputs": {
                    "targets": "targets",
                    "lstm_hidden_state": "h_in",
                    "lstm_cell_state": "c_in",
                },
                "outputs": {
                    "outputs": "decoder_output",
                    "lstm_hidden_state": "h_out",
                    "lstm_cell_state": "c_out",
                },
            },
            "joiner": {
                "filename": "joint.onnx",
                "inputs": {
                    "encoder_outputs": "encoder_output",
                    "decoder_outputs": "decoder_output",
                },
                "outputs": {
                    "logits": "joint_output",
                },
            },
            "vad": {
                "filename": "silero_vad.onnx",
                "threshold": 0.3,
                "silence_duration_ms": 3360,
                "prefix_padding_ms": 560,
            },
        },
    }

    with open(dst / "genai_config.json", "w") as f:
        json.dump(genai_config, f, indent=2)
    print(f"  [OK] genai_config.json")

    # Audio processor config
    window_size = preprocessor_cfg.get('window_size', preprocessor_cfg.get('n_window_size', 0.025))
    window_stride = preprocessor_cfg.get('window_stride', preprocessor_cfg.get('n_window_stride', 0.01))
    if isinstance(window_size, float) and window_size < 1.0:
        window_length_samples = int(window_size * sample_rate)
    elif isinstance(window_size, int):
        window_length_samples = window_size
    else:
        window_length_samples = 400
    if isinstance(window_stride, float) and window_stride < 1.0:
        hop_length_samples = int(window_stride * sample_rate)
    elif isinstance(window_stride, int):
        hop_length_samples = window_stride
    else:
        hop_length_samples = 160

    audio_config = {
        "model_type": "speech_features",
        "audio_params": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length_samples,
            "n_mels": n_mels,
            "window_length": window_length_samples,
            "window_type": "hann",
            "fmin": 0,
            "fmax": sample_rate // 2,
            "dither": preprocessor_cfg.get('dither', 0.0),
            "preemphasis": preemph,
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 1e-10,
            "normalize": preprocessor_cfg.get('normalize', 'none'),
            "center": True,
            "mag_power": 2.0,
        },
    }

    with open(dst / "audio_processor_config.json", "w") as f:
        json.dump(audio_config, f, indent=2)
    print(f"  [OK] audio_processor_config.json")
    print()


def download_silero_vad(output_dir: str):
    """Download the Silero VAD ONNX model from onnx-community/silero-vad."""
    from huggingface_hub import hf_hub_download

    print("=== Stage 6: Downloading Silero VAD ===")
    dst_dir = _resolve(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "silero_vad.onnx"

    cached = hf_hub_download(
        repo_id="onnx-community/silero-vad",
        filename="onnx/model.onnx",
    )
    shutil.copy2(cached, str(dst))
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  Saved Silero VAD model to: {dst} ({size_mb:.1f} MB)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Nemotron Speech Streaming for CPU inference"
    )
    parser.add_argument(
        "--model-name",
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="HuggingFace model name or path to a local .nemo file",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for optimized models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=0.56,
        choices=[0.08, 0.16, 0.56, 1.12],
        help="Streaming chunk size in seconds (default: 0.56)",
    )
    args = parser.parse_args()

    # Stages 1-3: Run Olive pipelines for encoder, decoder, joint
    run_olive_pipelines(output_dir=args.output_dir)

    # Stage 4: Export tokenizer
    run_tokenizer_export(model_name=args.model_name, output_dir=args.output_dir)

    # Stage 5: Generate config files
    generate_configs(
        model_name=args.model_name,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )

    # Stage 6: Download Silero VAD
    try:
        download_silero_vad(output_dir=args.output_dir)
    except Exception as exc:
        print(
            f"  Warning: Silero VAD download failed ({exc}).\n"
            "  You can download it manually later."
        )

    # Summary
    output_path = _resolve(args.output_dir)
    if output_path.exists():
        files = sorted(f for f in output_path.iterdir() if f.is_file())
        total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"=== Done! Optimized models → {output_path} ===")
        print(f"    Total size: {total_mb:.1f} MB")
        for f in files:
            tag = " ← INT4 k-quant (Olive)" if f.name.startswith("encoder") else ""
            print(f"    {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB){tag}")


if __name__ == "__main__":
    main()
