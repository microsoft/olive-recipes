"""End-to-end optimization pipeline for Gemma 4 ONNX models.

Builds four sub-models (decoder, vision_encoder, audio_encoder, embedding)
via MobiusModelBuilder, optionally applies INT4 quantization, and validates
the output GenAI package.

Usage:
    python optimize.py --device cpu
    python optimize.py --device gpu
    python optimize.py --device gpu --variant int4
"""

import argparse
import json
import sys
from pathlib import Path


def export_models(config_path: str):
    """Run Olive pipeline from a config file."""
    from olive import run

    print(f"=== Running Olive pipeline: {config_path} ===")
    run(config_path)
    print()


def validate_output(models_dir: str):
    """Validate that the output directory contains a valid GenAI package."""
    models_path = Path(models_dir)

    # Check required components
    expected_components = ["decoder", "vision_encoder", "audio_encoder", "embedding"]
    missing = []
    for component in expected_components:
        component_dir = models_path / component
        onnx_file = component_dir / "model.onnx"
        if not onnx_file.exists():
            missing.append(str(onnx_file))

    if missing:
        print(f"WARNING: Missing expected ONNX files: {missing}")
    else:
        print(f"  All {len(expected_components)} components present")

    # Check GenAI config
    genai_config_path = models_path / "genai_config.json"
    if not genai_config_path.exists():
        print("WARNING: genai_config.json not found")
        return False

    with open(genai_config_path) as f:
        config = json.load(f)

    model_config = config.get("model", {})

    # Validate model type
    model_type = model_config.get("type")
    if model_type != "gemma4":
        print(f"WARNING: Expected model type 'gemma4', got '{model_type}'")

    # Validate multimodal sections
    for section in ("decoder", "embedding", "vision", "speech"):
        if section not in model_config:
            print(f"WARNING: Missing '{section}' section in genai_config.json")

    # Validate special token IDs
    for token in ("image_token_id", "audio_token_id", "boa_token_id"):
        if token not in model_config:
            print(f"WARNING: Missing '{token}' in genai_config.json")

    # Check tokenizer
    if not (models_path / "tokenizer.json").exists():
        print("WARNING: tokenizer.json not found")

    # Check image processor
    if not (models_path / "image_processor.json").exists():
        print("WARNING: image_processor.json not found")

    print(f"  genai_config.json validated (model_type={model_type})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Optimize Gemma 4 ONNX models")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Target device",
    )
    parser.add_argument(
        "--variant",
        choices=["fp32", "fp16", "int4"],
        default=None,
        help="Model variant. Defaults: cpu=fp32, gpu=int4",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export, only validate existing output",
    )
    args = parser.parse_args()

    # Resolve config path
    if args.device == "cpu":
        config_dir = "cpu"
        variant = args.variant or "fp32"
        if variant != "fp32":
            print(f"ERROR: CPU only supports fp32 variant, got {variant}")
            sys.exit(1)
        config_path = f"{config_dir}/config.json"
        models_dir = f"{config_dir}/models"
    else:
        variant = args.variant or "int4"
        if variant not in ("fp16", "int4"):
            print(f"ERROR: GPU supports fp16 or int4 variants, got {variant}")
            sys.exit(1)
        config_path = f"cuda/{variant}/config.json"
        models_dir = f"cuda/{variant}/models"

    if not args.skip_export:
        export_models(config_path)

    print("=== Validating output ===")
    if validate_output(models_dir):
        print(f"\nDone. Output: {models_dir}/")
    else:
        print("\nWARNING: Validation found issues. Check output manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
