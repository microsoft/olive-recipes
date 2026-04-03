"""End-to-end optimization pipeline for Nemotron Speech Streaming.

Stage 1 — Export (NeMo → ONNX):
    Runs scripts/export_nemotron_to_onnx_static_shape.py to produce encoder,
    decoder, joint network, and config files from the NeMo model.
    Then runs scripts/export_tokenizer.py to produce tokenizer files
    (tokenizer.json, tokenizer_config.json, vocab.txt) in the same directory.

Stage 2 — Olive Graph Fusion (ONNX encoder → fused ONNX):
    Runs the Olive pipeline defined in encoder.json:
    - OrtTransformersOptimization (model_type="conformer"):
        Fuses Conformer subgraphs into SkipLayerNormalization, BiasGelu, etc.
        Multi-head attention fusion is disabled to avoid accuracy regression.

Stage 3 — INT4 k-quant Quantization (fused encoder → INT4 ONNX):
    Applies k-quant weight-only quantization (block_size=32, symmetric,
    accuracy_level=4) via OnnxRuntime's MatMulNBitsQuantizer, producing
    an INT4-only encoder.onnx.
    The decoder and joint networks remain FP32 and are copied unchanged.

Usage:
    # Full pipeline: export + Olive graph fusion + INT4 k-quant
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
_TOKENIZER_SCRIPT = _SCRIPT_DIR.parent / "scripts" / "export_tokenizer.py"

DEFAULT_EXPORT_DIR = "build/onnx_models_fp32"
DEFAULT_FUSED_DIR = "build/onnx_models_fused"
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


def run_olive_graph_fusion(export_dir: str, fused_dir: str):
    """Run Olive graph fusion on the encoder (OrtTransformersOptimization).

    Applies Conformer-specific fusions (SkipLayerNorm, BiasGelu, etc.) using
    the Olive pipeline defined in encoder.json.  Multi-head attention fusion is
    intentionally disabled to avoid an accuracy regression on this model.
    """
    from olive import run

    print("=== Stage 2: Olive Graph Fusion (encoder.json) ===")

    # Load the base config and patch in the actual export/fused paths so that
    # non-default --export-dir / --fused-dir values are honoured correctly.
    config_path = _SCRIPT_DIR / "encoder.json"
    with open(config_path) as f:
        config = json.load(f)

    config["input_model"]["model_path"] = str(_resolve(export_dir) / "encoder.onnx")
    config["output_dir"] = str(_resolve(fused_dir))

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


def run_kquant_quantization(
    fused_dir: str,
    output_dir: str,
    block_size: int = 32,
    is_symmetric: bool = True,
    accuracy_level: int = 4,
):
    """Quantize the fused encoder to INT4 using k-quant (KQuantWeightOnlyQuantConfig).

    Produces an INT4-only encoder with all MatMul weights quantized via the
    k-quant algorithm from OnnxRuntime's MatMulNBitsQuantizer.
    """
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        KQuantWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    print("=== Stage 3: INT4 k-quant Quantization ===")

    # Olive saves its output as "model.onnx"; a manually-fused dir may have
    # "encoder.onnx".  Check the Olive default first, then fall back.
    fused_path = _resolve(fused_dir)
    src = fused_path / "model.onnx"
    if not src.exists():
        src = fused_path / "encoder.onnx"
    if not src.exists():
        raise FileNotFoundError(
            f"Fused encoder not found in {fused_path}. "
            "Expected 'model.onnx' (Olive output) or 'encoder.onnx'."
        )
    dst_dir = _resolve(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "encoder.onnx"

    print(f"  Input:          {src}")
    print(f"  Output:         {dst}")
    print(f"  Bits:           4")
    print(f"  Block size:     {block_size}")
    print(f"  Symmetric:      {is_symmetric}")
    print(f"  Accuracy level: {accuracy_level}")
    print(f"  Algorithm:      k_quant (INT4-only)")

    model = onnx.load(str(src), load_external_data=True)
    algo_config = KQuantWeightOnlyQuantConfig()
    quantizer = MatMulNBitsQuantizer(
        model=model,
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=accuracy_level,
        algo_config=algo_config,
    )
    quantizer.process()

    for stale in [dst, Path(str(dst) + ".data")]:
        if stale.exists():
            stale.unlink()

    quantizer.model.save_model_to_file(str(dst), use_external_data_format=True)
    print(f"  Saved INT4 encoder to: {dst}")
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
        "--fused-dir",
        default=DEFAULT_FUSED_DIR,
        help=f"Directory for Olive graph-fused ONNX models (default: {DEFAULT_FUSED_DIR})",
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

    # Stage 2: Olive graph fusion (OrtTransformersOptimization, MHA fusion disabled)
    run_olive_graph_fusion(export_dir=args.export_dir, fused_dir=args.fused_dir)

    # Stage 3: INT4 k-quant quantization on fused encoder
    run_kquant_quantization(fused_dir=args.fused_dir, output_dir=args.output_dir)

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
