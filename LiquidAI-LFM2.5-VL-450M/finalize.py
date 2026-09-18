"""Wire the decoder, vision encoder and embedding model into one ONNX Runtime GenAI package.

The decoder recipe writes ``model.onnx``, ``genai_config.json`` and the tokenizer files to the
output directory; the vision recipe adds ``vision_encoder/`` and ``embedding/`` next to them.
This script adds the ``embedding`` and ``vision`` sections to ``genai_config.json`` and writes the
``processor_config.json`` that drives image preprocessing in ONNX Runtime GenAI.

Usage:
    python finalize.py [model_dir]
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

VISION_FILENAME = "vision_encoder/model.onnx"
EMBEDDING_FILENAME = "embedding/model.onnx"

# PIL resampling codes used by the Hugging Face image processor -> onnxruntime-extensions names.
INTERPOLATION = {2: "LINEAR", 3: "CUBIC"}

# LFM2.5-VL-3B pre-tokenizes with a Llama-3 style pattern whose leading `'(?i:...)` group the
# tokenizer in onnxruntime-extensions rejects ("Invalid '(?...)' zero-width assertion"). The
# pattern the other LFM2.5 models ship selects the same contractions and was checked to produce
# identical tokens on 20k random strings, chat prompts and code.
UNSUPPORTED_PRETOKENIZER_REGEX = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s"
)
SUPPORTED_PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def load_hf_json(filename: str) -> dict:
    return json.loads(Path(hf_hub_download(MODEL_ID, filename)).read_text())


def interpolation(resample: int) -> str:
    try:
        return INTERPOLATION[resample]
    except KeyError:
        raise SystemExit(f"unsupported resample {resample}: expected 2 (LINEAR) or 3 (CUBIC)") from None


def make_processor_config(image_processor: dict) -> dict:
    """Mirror the Hugging Face ``Lfm2VlImageProcessor``, minus image splitting (tiling)."""
    patch_size = image_processor["encoder_patch_size"]
    merge_size = image_processor["downsample_factor"]
    pixels_per_token = (patch_size * merge_size) ** 2
    return {
        "processor": {
            "name": "lfm2_vl_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
                {
                    "operation": {
                        "name": "resize",
                        "type": "Resize",
                        "attrs": {
                            "height": image_processor["size"]["height"],
                            "width": image_processor["size"]["width"],
                            "interpolation": interpolation(image_processor["resample"]),
                            "smart_resize": 1,
                            "min_pixels": image_processor["min_image_tokens"] * pixels_per_token,
                            "max_pixels": image_processor["max_image_tokens"] * pixels_per_token,
                            "patch_size": patch_size,
                            "merge_size": merge_size,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "rescale",
                        "type": "Rescale",
                        "attrs": {"rescale_factor": image_processor["rescale_factor"]},
                    }
                },
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {"mean": image_processor["image_mean"], "std": image_processor["image_std"]},
                    }
                },
                {"operation": {"name": "to_channel_first", "type": "Permute3D", "attrs": {"dims": [2, 0, 1]}}},
                {"operation": {"name": "image_sizes", "type": "PixtralImageSizes"}},
            ],
        }
    }


def fix_tokenizer_regex(tokenizer_path: Path) -> bool:
    """Swap the pre-tokenizer pattern onnxruntime-extensions cannot parse for its equivalent."""
    tokenizer = json.loads(tokenizer_path.read_text())
    pre_tokenizers = tokenizer.get("pre_tokenizer", {}).get("pretokenizers", [])
    changed = False
    for pre_tokenizer in pre_tokenizers:
        pattern = pre_tokenizer.get("pattern", {})
        if pattern.get("Regex") == UNSUPPORTED_PRETOKENIZER_REGEX:
            pattern["Regex"] = SUPPORTED_PRETOKENIZER_REGEX
            changed = True
    if changed:
        tokenizer_path.write_text(json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", nargs="?", default="model", help="output_dir of the recipes (default: model)")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    for filename in ("genai_config.json", "config.json", VISION_FILENAME, EMBEDDING_FILENAME):
        if not (model_dir / filename).exists():
            raise SystemExit(f"{model_dir / filename} not found: run both the decoder and the vision recipe first")

    genai_config_path = model_dir / "genai_config.json"
    genai_config = json.loads(genai_config_path.read_text())
    model = genai_config["model"]
    if model["type"] != "lfm2_vl":
        raise SystemExit(f'expected model type "lfm2_vl" in {genai_config_path}, got "{model["type"]}"')

    # config.json ships inside the package, so it always matches the weights that were built;
    # only the image processor settings still have to come from the Hub.
    hf_config = json.loads((model_dir / "config.json").read_text())
    image_processor = load_hf_json("processor_config.json")["image_processor"]

    # The vision and embedding sessions run on the same execution provider as the decoder.
    provider_options = model["decoder"]["session_options"]["provider_options"]
    session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}
    # Exception: the vision tower resizes the SigLIP2 position embeddings with an antialiased
    # Resize, which the WebGPU EP does not implement ("The antialias attribute of Resize operator
    # is NOT implemented"), so that one session falls back to CPU. Drop this override once the EP
    # supports it; the rest of the pipeline stays on WebGPU either way.
    on_webgpu = any(key.lower() == "webgpu" for option in provider_options for key in option)
    vision_session_options = {"log_id": "onnxruntime-genai", "provider_options": [] if on_webgpu else provider_options}
    model["embedding"] = {
        "filename": EMBEDDING_FILENAME,
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": session_options,
    }
    model["vision"] = {
        "filename": VISION_FILENAME,
        "config_filename": "processor_config.json",
        "patch_size": hf_config["encoder_patch_size"],
        "spatial_merge_size": hf_config["downsample_factor"],
        # Sequence length every image is padded to so several images can share one vision run.
        "max_num_patches": image_processor["max_num_patches"],
        "inputs": {
            "pixel_values": "pixel_values",
            "attention_mask": "pixel_attention_mask",
            "image_sizes": "spatial_shapes",
        },
        "outputs": {"image_features": "image_features"},
        "session_options": vision_session_options,
    }
    genai_config_path.write_text(json.dumps(genai_config, indent=4) + "\n")
    print(f"Updated {genai_config_path}")

    processor_config_path = model_dir / "processor_config.json"
    processor_config_path.write_text(json.dumps(make_processor_config(image_processor), indent=4) + "\n")
    print(f"Wrote {processor_config_path}")

    tokenizer_path = model_dir / "tokenizer.json"
    if fix_tokenizer_regex(tokenizer_path):
        print(f"Replaced the unsupported pre-tokenizer pattern in {tokenizer_path}")


if __name__ == "__main__":
    main()
