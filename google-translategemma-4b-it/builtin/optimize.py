"""End-to-end optimization pipeline for TranslateGemma-4B-IT VLM.

Exports three sub-models (text decoder, vision encoder, embedding),
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
    """Run Olive for all 3 sub-models (text, embedding, vision)."""
    from olive import run

    config_path = Path(config_dir)
    print(f"=== Running Olive pipelines (configs from {config_path}) ===")
    for config in ("text.json", "embedding.json", "vision.json"):
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
        "inputs": {"pixel_values": "pixel_values"},
        "outputs": {"image_features": "image_features"},
        "session_options": session_options,
    }

    # Force VLM type (needed when text model is Gemma3ForCausalLM which sets "gemma3_text")
    config["model"]["type"] = "gemma3"

    # TranslateGemma token IDs (from model/config.json)
    config["model"]["image_token_id"] = 262144
    config["model"]["bos_token_id"] = 2
    config["model"]["eos_token_id"] = [1, 106]
    config["model"]["pad_token_id"] = 0

    config["search"]["max_length"] = 2048
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 0.95

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    # Gemma3 image preprocessing: resize to 896x896, rescale 1/255, normalize [-1,1], HWC -> CHW
    processor_config = {
        "processor": {
            "name": "gemma_3_image_processing",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "resize", "type": "Resize", "attrs": {
                    "interpolation": "CUBIC", "width": 896, "height": 896, "keep_aspect_ratio": 0,
                }}},
                {"operation": {"name": "re-scale", "type": "Rescale"}},
                {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
                    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                }}},
                {"operation": {"name": "to_channel_first", "type": "Permute3D", "attrs": {
                    "dims": [2, 0, 1],
                }}},
            ],
        }
    }

    processor_path = Path(output_dir) / "processor_config.json"
    with open(processor_path, "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {processor_path}")


def fix_tokenizer(output_dir: str = MODELS_DIR):
    """Copy and fix tokenizer files for C++ std::regex compatibility if needed."""
    tk_path = Path(output_dir) / "tokenizer.json"
    if not tk_path.exists():
        return

    tk = json.loads(tk_path.read_text(encoding="utf-8"))
    pt = tk.get("pre_tokenizer", {})
    if pt.get("type") == "Sequence":
        original_count = len(pt.get("pretokenizers", []))
        pt["pretokenizers"] = [s for s in pt["pretokenizers"] if s.get("type") == "ByteLevel"]
        for s in pt["pretokenizers"]:
            s["use_regex"] = True
        new_count = len(pt["pretokenizers"])
        if new_count != original_count:
            tk_path.write_text(json.dumps(tk, ensure_ascii=False), encoding="utf-8")
            print(f"  Fixed tokenizer for C++ std::regex compatibility")


def main():
    parser = argparse.ArgumentParser(description="Optimize TranslateGemma-4B-IT VLM")
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
