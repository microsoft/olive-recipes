"""AITK aitkpython driver for the Qwen3.5-2B NVIDIA TRT for RTX recipe.

The Qwen3.5 vision-language model is exported as three ONNX sub-models
(vision encoder, text embedding, text decoder). AITK only runs a single
Olive workflow per recipe, so this script wraps the three inner Olive
configs behind one ``AitkPython`` pass: it runs each inner config, then
patches ``genai_config.json``/``processor_config.json`` and the tokenizer
for the ONNX Runtime GenAI runtime.

Adapted from ``../builtin/optimize.py`` (CUDA recipe) using the
``NvTensorRTRTXExecutionProvider`` execution provider.
"""

import argparse
import json
import logging
import os

import olive.workflows

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)
logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INNER_CONFIGS = {
    "trtrtx_embedding.json": "embedding.onnx",
    "trtrtx_text.json": "text.onnx",
    "trtrtx_vision.json": "vision.onnx",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to input config file")
    parser.add_argument("--model_config", help="path to input model config file")
    parser.add_argument("--runtime", required=True, help="runtime")
    return parser.parse_args()


def load_update_config(config_path: str, cache_dir: str, output_dir: str, output_name: str) -> dict:
    """Load an inner Olive config and rewire its cache/output dirs."""
    with open(config_path, "r", encoding="utf-8") as f:
        oliveJson = json.load(f)

    oliveJson["cache_dir"] = cache_dir
    # all sub-models land in the single genai model folder
    oliveJson["output_dir"] = os.path.join(output_dir, output_name) if output_name else output_dir
    return oliveJson


def copy_olive_config(history_folder: str, config_name: str, cache_dir: str, output_dir: str, output_name: str) -> dict:
    """Save the resolved inner config into the history folder for record, and return it."""
    logger.info(f"Copying {config_name} to {history_folder}...")
    oliveJson = load_update_config(os.path.join(SCRIPT_DIR, config_name), cache_dir, output_dir, output_name)
    os.makedirs(history_folder, exist_ok=True)
    with open(os.path.join(history_folder, config_name), "w", encoding="utf-8") as f:
        json.dump(oliveJson, f, indent=4)
    return oliveJson


def export_models(history_folder: str, models_dir: str, cache_dir: str):
    """Run Olive for all 3 sub-models (embedding, text, vision) into models_dir."""
    for config_name in INNER_CONFIGS:
        oliveJson = copy_olive_config(history_folder, config_name, cache_dir, models_dir, INNER_CONFIGS[config_name])
        logger.info(f"Running {config_name}...")
        output = olive.workflows.run(oliveJson)
        if output is None or (hasattr(output, "has_output_model") and not output.has_output_model()):
            raise Exception(f"Model file is not generated for {config_name}")


def update_genai_config(models_dir: str, device: str = "gpu"):
    """Patch genai_config.json with embedding/vision sections and processor_config."""
    config_path = os.path.join(models_dir, "genai_config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if device == "gpu":
        provider_options = [{"cuda": {"enable_cuda_graph": "1", "enable_skip_layer_norm_strict_mode": "1"}}]
        # Vision model has Loop nodes (one per ViT block) which are incompatible
        # with CUDA graph capture, so disable it for vision and embedding only.
        vision_provider_options = [{"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1"}}]
    else:
        provider_options = []
        vision_provider_options = []

    session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}
    vision_session_options = {"log_id": "onnxruntime-genai", "provider_options": vision_provider_options}

    config["model"]["decoder"]["session_options"] = session_options

    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": vision_session_options,
    }

    config["model"]["vision"] = {
        "filename": "vision.onnx",
        "config_filename": "processor_config.json",
        "spatial_merge_size": 2,
        "tokens_per_second": 2.0,
        "patch_size": 16,
        "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        "outputs": {"image_features": "image_features"},
        "session_options": vision_session_options,
    }

    config["model"]["bos_token_id"] = 248044
    config["model"]["eos_token_id"] = [248044]
    config["model"]["pad_token_id"] = 248044
    config["model"]["image_token_id"] = 248056
    config["model"]["video_token_id"] = 248057
    config["model"]["vision_start_token_id"] = 248053

    config["search"]["top_k"] = 1
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Updated {config_path}")

    processor_config = {
        "processor": {
            "name": "qwen2_5_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {
                    "operation": {
                        "name": "resize",
                        "type": "Resize",
                        "attrs": {
                            "width": 960,
                            "height": 672,
                            "smart_resize": 1,
                            "min_pixels": 65536,
                            "max_pixels": 16777216,
                            "patch_size": 16,
                            "merge_size": 2,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "rescale",
                        "type": "Rescale",
                        "attrs": {
                            "rescale_factor": 0.00392156862745098,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                            "qwen2_5_vl": 1,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "patch_image",
                        "type": "PatchImage",
                        "attrs": {
                            "patch_size": 16,
                            "temporal_patch_size": 2,
                            "merge_size": 2,
                        },
                    }
                },
            ],
        }
    }

    processor_path = os.path.join(models_dir, "processor_config.json")
    with open(processor_path, "w", encoding="utf-8") as f:
        json.dump(processor_config, f, indent=2)
    logger.info(f"Created {processor_path}")


def fix_tokenizer(models_dir: str):
    """Fix tokenizer.json for C++ std::regex compatibility.

    Qwen3.5's tokenizer uses Unicode property escapes (\\p{L}, \\p{N}) in its
    Split pre-tokenizer, which aren't supported by std::regex in onnxruntime-genai.
    Remove the Split and keep only ByteLevel with use_regex=True.
    """
    tk_path = os.path.join(models_dir, "tokenizer.json")
    if not os.path.exists(tk_path):
        return
    with open(tk_path, "r", encoding="utf-8") as f:
        tk = json.load(f)
    pt = tk.get("pre_tokenizer", {})
    if pt.get("type") == "Sequence":
        pt["pretokenizers"] = [s for s in pt["pretokenizers"] if s.get("type") == "ByteLevel"]
        for s in pt["pretokenizers"]:
            s["use_regex"] = True
    with open(tk_path, "w", encoding="utf-8") as f:
        json.dump(tk, f, ensure_ascii=False)

    tc_path = os.path.join(models_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        tc["tokenizer_class"] = "Qwen2Tokenizer"
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
    logger.info("Fixed tokenizer for C++ std::regex compatibility")


def main():
    args = parse_arguments()

    with open(args.config, "r", encoding="utf-8") as f:
        olive_json = json.load(f)

    # evaluation entrypoint — this recipe has no evaluation.
    if args.model_config:
        return

    history_folder = os.path.dirname(args.config)
    models_dir = olive_json["output_dir"]
    cache_dir = olive_json["cache_dir"]
    os.makedirs(models_dir, exist_ok=True)

    import sys

    # A fix for Windows console output encoding issues (e.g., encodings\cp1252.py)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    export_models(history_folder, models_dir, cache_dir)

    logger.info("=== Generating genai configs ===")
    update_genai_config(models_dir, device="gpu")
    fix_tokenizer(models_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
