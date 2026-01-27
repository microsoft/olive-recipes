import argparse
import json
import logging
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)


def update_genai_config(output_dir: str = "models", device: str = "gpu"):
    config_path = Path(output_dir) / "genai_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Set provider_options based on device
    if device == "gpu":
        provider_options = [
            {
                "cuda": {
                    "enable_cuda_graph": "0",
                    "enable_skip_layer_norm_strict_mode": "1"
                }
            }
        ]
    else:
        provider_options = []

    # Add embedding configuration
    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {
            "input_ids": "input_ids",
            "image_features": "image_features"
        },
        "outputs": {
            "inputs_embeds": "inputs_embeds"
        },
        "session_options": {
            "log_id": "onnxruntime-genai",
            "provider_options": provider_options
        }
    }

    # Add vision configuration
    config["model"]["vision"] = {
        "filename": "vision.onnx",
        "config_filename": "processor_config.json",
        "spatial_merge_size": 2,
        "tokens_per_second": 2.0,
        "inputs": {
            "pixel_values": "pixel_values",
            "image_grid_thw": "image_grid_thw"
        },
        "outputs": {
            "image_features": "image_features"
        },
        "session_options": {
            "log_id": "onnxruntime-genai",
            "provider_options": provider_options
        }
    }

    # Add required token IDs for vision
    config["model"]["image_token_id"] = 151655
    config["model"]["video_token_id"] = 151656
    config["model"]["vision_start_token_id"] = 151652

    # Fix top_k and top_p if they are null
    if config["search"].get("top_k") is None:
        config["search"]["top_k"] = 50
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Updated genai_config.json at {config_path}")

    # Create processor_config.json for image processing
    processor_config = {
        "processor": {
            "name": "qwen2_5_image_processor",
            "transforms": [
                {
                    "operation": {
                        "name": "decode_image",
                        "type": "DecodeImage",
                        "attrs": {
                            "color_space": "RGB"
                        }
                    }
                },
                {
                    "operation": {
                        "name": "convert_to_rgb",
                        "type": "ConvertRGB"
                    }
                },
                {
                    "operation": {
                        "name": "resize",
                        "type": "Resize",
                        "attrs": {
                            "width": 540,
                            "height": 360,
                            "smart_resize": 1,
                            "min_pixels": 3136,
                            "max_pixels": 12845056,
                            "patch_size": 14,
                            "merge_size": 2
                        }
                    }
                },
                {
                    "operation": {
                        "name": "rescale",
                        "type": "Rescale",
                        "attrs": {
                            "rescale_factor": 0.00392156862745098
                        }
                    }
                },
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {
                            "mean": [0.48145466, 0.4578275, 0.40821073],
                            "std": [0.26862954, 0.26130258, 0.27577711],
                            "qwen2_5_vl": 1
                        }
                    }
                },
                {
                    "operation": {
                        "name": "patch_image",
                        "type": "PatchImage",
                        "attrs": {
                            "patch_size": 14,
                            "temporal_patch_size": 2,
                            "merge_size": 2
                        }
                    }
                }
            ]
        }
    }

    processor_config_path = Path(output_dir) / "processor_config.json"
    with open(processor_config_path, "w") as f:
        json.dump(processor_config, f, indent=2)

    print(f"Created processor_config.json at {processor_config_path}")


def optimize(device: str = "gpu"):
    print(f"Optimizing the model for {device.upper()}...")
    from olive import run

    run("embedding.json")
    run("text.json")
    run("vision.json")

    update_genai_config(device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Qwen2.5-VL model with Olive")
    parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Target device for inference (default: gpu)"
    )
    args = parser.parse_args()
    optimize(device=args.device)
