"""End-to-end optimization pipeline for Qwen3.5 ONNX models.

Exports three sub-models (vision encoder, text embedding, text decoder),
applies graph optimizations and INT4 quantization via Olive passes.

Usage:
    python optimize.py --config-dir cpu_and_mobile --device cpu
    python optimize.py --config-dir cpu_and_mobile --device cpu --skip-export
"""
import argparse
import json
import logging
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODELS_DIR = "models"


def export_models(config_dir: str):
    """Run Olive for all 3 sub-models (embedding, text, vision)."""
    from olive import run

    config_path = Path(config_dir)
    print(f"=== Running Olive pipelines (configs from {config_path}) ===")
    for config in ("embedding.json", "text.json", "vision.json"):
        print(f"  Running {config}...")
        run(str(config_path / config))
    print()


def update_genai_config(output_dir: str = MODELS_DIR, device: str = "cpu"):
    """Patch genai_config.json with embedding/vision sections and processor_config."""
    config_path = Path(output_dir) / "genai_config.json"

    with open(config_path) as f:
        config = json.load(f)

    if device == "gpu":
        provider_options = [
            {"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1"}}
        ]
    else:
        provider_options = []

    session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}

    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": session_options,
    }

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

    config["model"]["image_token_id"] = 248056
    config["model"]["video_token_id"] = 248057
    config["model"]["vision_start_token_id"] = 248053

    if config["search"].get("top_k") is None:
        config["search"]["top_k"] = 50
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

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


def fix_tokenizer(output_dir: str = MODELS_DIR):
    """Fix tokenizer.json for C++ std::regex compatibility.

    Qwen3.5's tokenizer uses Unicode property escapes (\\p{L}, \\p{N}) in its
    Split pre-tokenizer, which aren't supported by std::regex in onnxruntime-genai.
    Remove the Split and keep only ByteLevel with use_regex=True.
    """
    tk_path = Path(output_dir) / "tokenizer.json"
    if not tk_path.exists():
        return
    tk = json.loads(tk_path.read_text(encoding="utf-8"))
    pt = tk.get("pre_tokenizer", {})
    if pt.get("type") == "Sequence":
        pt["pretokenizers"] = [s for s in pt["pretokenizers"] if s.get("type") == "ByteLevel"]
        for s in pt["pretokenizers"]:
            s["use_regex"] = True
    tk_path.write_text(json.dumps(tk, ensure_ascii=False), encoding="utf-8")

    tc_path = Path(output_dir) / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text(encoding="utf-8"))
        tc["tokenizer_class"] = "Qwen2Tokenizer"
        tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Fixed tokenizer for C++ std::regex compatibility")


def main():
    parser = argparse.ArgumentParser(description="Optimize Qwen3.5 ONNX models")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--config-dir", default="cpu_and_mobile")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    models_dir = args.models_dir or str(Path(args.config_dir) / MODELS_DIR)

    if not args.skip_export:
        export_models(args.config_dir)

    print("=== Generating configs ===")
    update_genai_config(output_dir=models_dir, device=args.device)
    fix_tokenizer(output_dir=models_dir)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
