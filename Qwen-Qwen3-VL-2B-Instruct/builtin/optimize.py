"""End-to-end optimization pipeline for Qwen3-VL ONNX models.

All ONNX graph transformations (Gemm→MatMul, Cast chain elimination,
INT4 quantization) are now handled by Olive passes declared in the JSON
configs.  This script orchestrates the three Olive runs and writes the
GenAI runtime configuration files.

Usage:
    # Full pipeline: export + optimize + INT4 quantize (CPU)
    python optimize.py --config-dir cpu_and_mobile --device cpu

    # CUDA pipeline
    python optimize.py --config-dir cuda --device gpu

    # Skip export (models already exist, just regenerate configs)
    python optimize.py --config-dir cpu_and_mobile --device cpu --skip-export
"""
import argparse
import json
import logging
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODELS_DIR = "models"


# =============================================================================
# 1. Olive Export + Optimization + Quantization (all driven by JSON configs)
# =============================================================================

def export_models(config_dir: str):
    """Run Olive for all 3 sub-models (embedding, text, vision).

    The JSON configs define the full pipeline: export → graph surgeries
    → ORT optimization → Cast chain elimination → Gemm→MatMul → INT4
    quantization.
    """
    from olive import run

    config_path = Path(config_dir)
    print(f"=== Running Olive pipelines (configs from {config_path}) ===")
    for config in ("embedding.json", "text.json", "vision.json"):
        print(f"  Running {config}...")
        run(str(config_path / config))
    print()


# =============================================================================
# 2. GenAI Runtime Config Generation
# =============================================================================

def update_genai_config(output_dir: str = MODELS_DIR, device: str = "gpu"):
    """Patch genai_config.json with embedding/vision sections and processor_config."""
    config_path = Path(output_dir) / "genai_config.json"

    with open(config_path) as f:
        config = json.load(f)

    # Provider options
    if device == "gpu":
        provider_options = [
            {"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1"}}
        ]
    else:
        provider_options = []

    session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}

    # Embedding configuration
    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": session_options,
    }

    # Vision configuration
    config["model"]["vision"] = {
        "filename": "vision.onnx",
        "config_filename": "processor_config.json",
        "spatial_merge_size": 2,
        "tokens_per_second": 2.0,
        "patch_size": 16,
        "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        "outputs": {"image_features": "image_features"},
        "session_options": session_options,
    }

    config["model"]["image_token_id"] = 151655
    config["model"]["video_token_id"] = 151656
    config["model"]["vision_start_token_id"] = 151652

    # Fix null search params
    if config["search"].get("top_k") is None:
        config["search"]["top_k"] = 50
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    # Create processor_config.json (Qwen3-VL uses patch_size=16)
    processor_config = {
        "processor": {
            "name": "qwen3_vl_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {"operation": {"name": "resize", "type": "Resize", "attrs": {
                    "width": 540, "height": 360, "smart_resize": 1,
                    "min_pixels": 3136, "max_pixels": 12845056, "patch_size": 16, "merge_size": 2,
                }}},
                {"operation": {"name": "rescale", "type": "Rescale", "attrs": {
                    "rescale_factor": 0.00392156862745098,
                }}},
                {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
                    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "qwen3_vl": 1,
                }}},
                {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
                    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                }}},
            ],
        }
    }

    processor_path = Path(output_dir) / "processor_config.json"
    with open(processor_path, "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {processor_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimize Qwen3-VL ONNX models")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu",
                        help="Target device (default: cpu)")
    parser.add_argument("--config-dir", default="cpu_and_mobile",
                        help="Directory containing Olive JSON configs (default: cpu_and_mobile)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip Olive export (models already exist)")
    parser.add_argument("--models-dir", default=None,
                        help="Models directory (default: <config-dir>/models)")
    args = parser.parse_args()

    models_dir = args.models_dir or str(Path(args.config_dir) / MODELS_DIR)

    # Step 1: Export + optimize + quantize (all in Olive JSON pipelines)
    if not args.skip_export:
        export_models(args.config_dir)

    # Step 2: Generate GenAI runtime configs
    print("=== Generating configs ===")
    update_genai_config(output_dir=models_dir, device=args.device)
    print()

    print("Done.")


if __name__ == "__main__":
    main()
