"""End-to-end optimization pipeline for Nemotron Speech Streaming.

Stage 1 — Export (NeMo → ONNX):
    Runs scripts/export_nemotron_to_onnx_static_shape.py to produce encoder,
    decoder, joint network, tokenizer, and config files from the NeMo model.

Stage 2 — Olive Optimization (ONNX encoder → INT4 ONNX):
    Runs the Olive pipeline defined in encoder.json:
    - OrtTransformersOptimization (model_type="conformer"):
        Fuses Conformer subgraphs into MultiHeadAttention, SkipLayerNormalization,
        BiasGelu, etc. for faster CPU inference.
    - OnnxBlockWiseRtnQuantization:
        INT4 RTN weight quantization (block_size=32, symmetric, accuracy_level=4).
    The decoder and joint networks remain FP32 and are copied unchanged.

Usage:
    # Full pipeline: export + Olive optimize
    python optimize.py

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
    print()


def run_olive_optimization(export_dir: str, output_dir: str):
    """Run Olive optimization (graph fusion + INT4 quantization) on the encoder."""
    from olive import run

    print("=== Stage 2: Olive Optimization (encoder.json) ===")

    # Load the base config and patch in the actual export/output paths so that
    # non-default --export-dir / --output-dir values are honoured correctly.
    config_path = _SCRIPT_DIR / "encoder.json"
    with open(config_path) as f:
        config = json.load(f)

    config["input_model"]["model_path"] = str(_resolve(export_dir) / "encoder.onnx")
    # Olive uses output_dir as the parent directory for the saved model file.
    # Following the Qwen3-VL convention, the path ends with the model filename
    # so that Olive writes encoder.onnx directly into the output directory.
    config["output_dir"] = str(_resolve(output_dir) / "encoder.onnx")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(_SCRIPT_DIR), delete=False
    ) as tmp:
        json.dump(config, tmp, indent=4)
        tmp_path = tmp.name

    try:
        run(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

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
        help="Number of left context chunks for streaming attention (default: 10)",
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

    # Stage 1: Export NeMo model → ONNX (encoder, decoder, joint, configs)
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

    # Stage 2: Olive optimization — graph fusion + INT4 quantization on encoder
    run_olive_optimization(export_dir=args.export_dir, output_dir=args.output_dir)

    # Copy decoder, joint, and config files to output (FP32, unchanged)
    print("=== Copying supporting files ===")
    copy_supporting_files(export_dir=args.export_dir, output_dir=args.output_dir)
    print()

    # Summary
    output_path = _resolve(args.output_dir)
    if output_path.exists():
        files = sorted(f for f in output_path.iterdir() if f.is_file())
        total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"=== Done! Optimized models → {output_path} ===")
        print(f"    Total size: {total_mb:.1f} MB")
        for f in files:
            tag = " ← encoder (INT4, Olive-optimized)" if f.name.startswith("encoder") else ""
            print(f"    {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB){tag}")


if __name__ == "__main__":
    main()
