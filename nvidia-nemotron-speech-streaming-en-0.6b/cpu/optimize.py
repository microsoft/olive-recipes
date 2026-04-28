"""End-to-end optimization pipeline for Nemotron Speech Streaming.

Stage 1 — Export (NeMo → ONNX):
    Runs scripts/export_nemotron_to_onnx_static_shape.py to produce encoder,
    decoder, joint network, and config files from the NeMo model.
    Then runs scripts/export_tokenizer.py to produce tokenizer files
    (tokenizer.json, tokenizer_config.json, vocab.txt) in the same directory.

Stage 2 — Olive Encoder Pipeline (PyTorch → fused INT4 ONNX):
    Runs the Olive workflow defined in nemotron_speech_int4_cpu.json:
    - OnnxConversion: Loads encoder via nemotron_encoder_load.py → ONNX
    - OrtTransformersOptimization (model_type="conformer"):
        Fuses Conformer subgraphs (SkipLayerNorm, BiasGelu, etc.).
        Multi-head attention fusion is disabled to avoid accuracy regression.
    - OnnxKQuantQuantization:
        INT4 k-quant quantization (block_size=32, asymmetric).

    The decoder and joint networks remain FP32 and are copied unchanged.

Usage:
    # Full pipeline: export + Olive encoder pipeline
    python optimize.py

    # Or use Olive CLI directly for the encoder only:
    python -m olive run --config cpu/nemotron_speech_int4_cpu.json

    # Skip NeMo export (models already in build/onnx_models_fp32/)
    python optimize.py --skip-export

    # Custom streaming chunk size
    python optimize.py --chunk-size 1.12 --left-chunks 10
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_EXPORT_SCRIPT = _SCRIPT_DIR.parent / "scripts" / "export_nemotron_to_onnx_static_shape.py"
_TOKENIZER_SCRIPT = _SCRIPT_DIR.parent / "scripts" / "export_tokenizer.py"

DEFAULT_EXPORT_DIR = "build/onnx_models_fp32"
DEFAULT_OUTPUT_DIR = "build/onnx_models_int4"


def _resolve(path: str) -> Path:
    """Resolve a path relative to the cpu/ directory."""
    p = Path(path)
    return p if p.is_absolute() else _SCRIPT_DIR / p


def run_export(
    model_name: str,
    export_dir: str,
    chunk_size: float,
    left_chunks: int,
    device: str,
):
    """Export NeMo model to ONNX using the custom export script."""
    print(f"=== Stage 1: Exporting {model_name} to ONNX ===")
    cmd = [
        sys.executable,
        str(_EXPORT_SCRIPT),
        "--model_name", model_name,
        "--output_dir", str(_resolve(export_dir)),
        "--streaming",
        "--chunk_size", str(chunk_size),
        "--left_chunks", str(left_chunks),
        "--device", device,
    ]
    result = subprocess.run(cmd, cwd=str(_SCRIPT_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"Export failed (exit code {result.returncode})")

    # Export tokenizer files (tokenizer.json, tokenizer_config.json, vocab.txt)
    # to the same directory so they are copied to the final output alongside
    # decoder, joint, and config files.
    print("=== Stage 1b: Exporting tokenizer files ===")
    tokenizer_cmd = [
        sys.executable,
        str(_TOKENIZER_SCRIPT),
        "--model_name", model_name,
        "--output_dir", str(_resolve(export_dir)),
    ]
    result = subprocess.run(tokenizer_cmd, check=True, cwd=str(_SCRIPT_DIR))
    print()


def run_olive_encoder_pipeline(output_dir: str):
    """Run the Olive pipeline for the encoder: convert → fusion → INT4 quantization.

    Uses nemotron_speech_int4_cpu.json which loads the model via
    nemotron_encoder_load.py (model_script) and applies built-in Olive passes:
    OnnxConversion → OrtTransformersOptimization → OnnxKQuantQuantization.
    """
    from olive import run as olive_run

    print("=== Stage 2: Olive Encoder Pipeline (convert → fusion → k_quant INT4) ===")

    config_path = _SCRIPT_DIR / "nemotron_speech_int4_cpu.json"
    with open(config_path) as f:
        config = json.load(f)

    config["output_dir"] = str(_resolve(output_dir) / "encoder.onnx")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(_SCRIPT_DIR), delete=False
    ) as tmp:
        json.dump(config, tmp, indent=4)
        tmp_path = tmp.name

    try:
        olive_run(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print()


def download_silero_vad(output_dir: str):
    """Download the Silero VAD ONNX model from onnx-community/silero-vad.

    The model is saved as silero_vad.onnx in the output directory alongside
    the other ONNX models so it is available for voice-activity detection
    during inference.
    """
    from huggingface_hub import hf_hub_download

    print("=== Downloading Silero VAD ONNX model ===")
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


def copy_supporting_files(export_dir: str, output_dir: str):
    """Copy decoder, joint, tokenizer, and config files to the output directory.

    The encoder is produced by Olive and is intentionally excluded here.
    All other files (decoder.onnx, joint.onnx, genai_config.json,
    audio_processor_config.json, tokenizer files) remain FP32 and are
    copied unchanged.
    """
    src = _resolve(export_dir)
    dst = _resolve(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_file in sorted(src.iterdir()):
        if not src_file.is_file():
            continue
        if src_file.name.startswith("encoder"):
            continue  # Encoder is produced by Olive; skip
        dst_file = dst / src_file.name
        if src_file.resolve() != dst_file.resolve():
            shutil.copy2(str(src_file), str(dst_file))
            copied += 1

    print(
        f"  Copied {copied} supporting files "
        f"(decoder, joint, configs, tokenizer) → {dst}"
    )


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
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        help=f"Directory for exported FP32 ONNX models (default: {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for optimized INT4 models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=0.56,
        choices=[0.08, 0.16, 0.56, 1.12],
        help="Streaming chunk size in seconds (default: 0.56)",
    )
    parser.add_argument(
        "--left-chunks",
        type=int,
        default=10,
        help="Number of left chunks to look back in streaming mode (default: 10). "
             "left_context = left_chunks * (chunk_mel_frames / subsampling_factor)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for NeMo model export (default: cpu)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip the NeMo export step (reuse models already in --export-dir)",
    )
    args = parser.parse_args()

    # Stage 1: Export NeMo model → ONNX (decoder, joint, configs, tokenizer)
    # The encoder is handled by Olive in Stage 2.
    if not args.skip_export:
        run_export(
            model_name=args.model_name,
            export_dir=args.export_dir,
            chunk_size=args.chunk_size,
            left_chunks=args.left_chunks,
            device=args.device,
        )
    else:
        print(f"=== Stage 1: Skipped (reusing models from {args.export_dir}) ===\n")

    # Stage 2: Olive encoder pipeline (convert → fusion → INT4 quantization)
    run_olive_encoder_pipeline(output_dir=args.output_dir)

    # Copy decoder, joint, and config files to output (FP32, unchanged)
    print("=== Copying supporting files ===")
    copy_supporting_files(export_dir=args.export_dir, output_dir=args.output_dir)
    print()

    # Download Silero VAD ONNX model alongside the other ONNX models
    try:
        download_silero_vad(output_dir=args.output_dir)
    except Exception as exc:
        print(
            f"  Warning: Silero VAD download failed ({exc}).\n"
            "  You can download it manually with:\n"
            "    hf download onnx-community/silero-vad --include onnx/model.onnx --local-dir .\n"
            "  or:\n"
            "    huggingface-cli download onnx-community/silero-vad --include onnx/model.onnx --local-dir .\n"
            f"  and copy onnx/model.onnx to {_resolve(args.output_dir) / 'silero_vad.onnx'}"
        )

    # Clean up intermediate export directory (only if we created it)
    if not args.skip_export:
        export_path = _resolve(args.export_dir)
        if export_path.exists() and export_path.resolve() != _resolve(args.output_dir).resolve():
            shutil.rmtree(str(export_path))
            print(f"=== Cleaned up intermediate files: {export_path} ===")

    # Summary
    output_path = _resolve(args.output_dir)
    if output_path.exists():
        files = sorted(f for f in output_path.iterdir() if f.is_file())
        total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"=== Done! Optimized models → {output_path} ===")
        print(f"    Total size: {total_mb:.1f} MB")
        for f in files:
            tag = " ← encoder (INT4 k-quant, Olive-optimized)" if f.name.startswith("encoder") else ""
            print(f"    {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB){tag}")


if __name__ == "__main__":
    main()
