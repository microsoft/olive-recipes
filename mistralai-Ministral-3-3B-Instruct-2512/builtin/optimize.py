"""End-to-end optimization pipeline for Ministral-3-3B ONNX models.

Uses mobius for vision and embedding export (reliable dynamo-free ONNX
construction), and Olive/ModelBuilder for text decoder export (GQA + INT4).

Usage:
    python optimize.py --config-dir cpu_and_mobile --device cpu
    python optimize.py --config-dir cuda --device gpu
    python optimize.py --config-dir cpu_and_mobile --device cpu --skip-export
    python optimize.py --config-dir cpu_and_mobile --device cpu --model-path /local/dequantized/checkpoint
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import user_script

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODELS_DIR = "models"
DEFAULT_HF_MODEL = user_script.MODEL_NAME


def _fix_vision_output_rank(onnx_path: str):
    """Squeeze batch dimension from vision model output if rank is 3.

    Mobius exports vision as [batch, num_patches, hidden_size] (rank 3), but
    the embedding model expects [num_patches, hidden_size] (rank 2). This adds
    a Squeeze(axis=0) node to the vision model's image_features output.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(onnx_path, load_external_data=False)
    output = model.graph.output[0]
    rank = len(output.type.tensor_type.shape.dim)

    if rank != 3:
        return

    print(f"  [PostProcess] Squeezing vision output from rank {rank} to rank 2")

    old_name = output.name
    intermediate_name = f"{old_name}_unsqueezed"

    # Rename the current output producer
    for node in model.graph.node:
        for i, out in enumerate(node.output):
            if out == old_name:
                node.output[i] = intermediate_name

    # Add Squeeze(axis=0) node
    axes_init = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [0])
    model.graph.initializer.append(axes_init)
    squeeze_node = helper.make_node(
        "Squeeze", [intermediate_name, "squeeze_axes"], [old_name], name="Squeeze_batch"
    )
    model.graph.node.append(squeeze_node)

    # Update output shape to rank 2 (drop batch dim)
    old_dims = list(output.type.tensor_type.shape.dim)
    del output.type.tensor_type.shape.dim[:]
    output.type.tensor_type.shape.dim.extend(old_dims[1:])

    onnx.save(model, onnx_path)
    print(f"  [PostProcess] Vision output squeezed: {old_name} is now rank 2")


def _fix_decoder_external_data(decoder_dir: str):
    """Fix external data filename if ONNX model references a different data file.

    ModelBuilder names the ONNX file based on the config (e.g., 'text.onnx') but
    we save it as 'model.onnx'. The external data reference still points to
    'text.onnx.data'. Fix by updating references and renaming the data file.
    """
    import onnx

    onnx_path = os.path.join(decoder_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        return

    model = onnx.load(onnx_path, load_external_data=False)
    data_refs = set()
    for init in model.graph.initializer:
        for ext in init.external_data:
            if ext.key == "location":
                data_refs.add(ext.value)

    needs_fix = False
    for ref in data_refs:
        if ref != "model.onnx.data":
            src = os.path.join(decoder_dir, ref)
            dst = os.path.join(decoder_dir, "model.onnx.data")
            if os.path.exists(src):
                os.replace(src, dst)
            needs_fix = True

    if needs_fix:
        for init in model.graph.initializer:
            for ext in init.external_data:
                if ext.key == "location" and ext.value != "model.onnx.data":
                    ext.value = "model.onnx.data"
        onnx.save(model, onnx_path)
        print("  [PostProcess] Fixed external data references → model.onnx.data")


def export_text_decoder(config_dir: str):
    """Export text decoder using Olive/ModelBuilder (GQA + quantization)."""
    try:
        from olive import run
    except ImportError:
        from olive.workflows import run

    config_path = Path(config_dir) / "text.json"
    if config_path.exists():
        print(f"  [Olive] Exporting text decoder from {config_path}...")
        run(str(config_path))
    else:
        raise FileNotFoundError(f"Text config not found: {config_path}")


def export_vision_and_embedding(
    output_dir: str,
    model_path: str,
    dtype: str = "f16",
):
    """Export vision encoder and embedding using mobius.

    Mobius constructs the ONNX graph declaratively and applies pretrained
    weights, avoiding torch.onnx.export dynamo issues with Pixtral's
    dynamic image dimensions.
    """
    from mobius import build

    print(f"  [Mobius] Building VLM from {model_path} (dtype={dtype})...")
    # mobius.build() accepts dtype as a string (e.g. "f16", "f32", "bf16")
    # and resolves it internally — pass the CLI string directly
    pkg = build(model_path, dtype=dtype, load_weights=True)

    os.makedirs(output_dir, exist_ok=True)

    required_components = ("vision", "embedding")
    missing_components = []

    for component in required_components:
        if component in pkg:
            component_dir = os.path.join(output_dir, component)
            os.makedirs(component_dir, exist_ok=True)
            try:
                pkg.save(
                    component_dir,
                    components=lambda name, c=component: name == c,
                    check_weights=True,
                )
                # mobius saves as model.onnx directly in component_dir
                expected_onnx = os.path.join(component_dir, "model.onnx")
                if not os.path.exists(expected_onnx):
                    raise FileNotFoundError(
                        f"Mobius export did not produce expected ONNX file for '{component}': {expected_onnx}"
                    )
                print(f"  [Mobius] Saved {expected_onnx}")
            except Exception:
                shutil.rmtree(component_dir, ignore_errors=True)
                raise
        else:
            missing_components.append(component)

    if missing_components:
        raise ValueError(
            "Mobius package is missing required component(s): "
            + ", ".join(missing_components)
        )

    print("  [Mobius] Vision and embedding export complete")

    # Post-process: fix vision output rank (batch dim squeeze)
    vision_onnx = os.path.join(output_dir, "vision", "model.onnx")
    if os.path.exists(vision_onnx):
        _fix_vision_output_rank(vision_onnx)


def _assemble_decoder(output_dir: str):
    """Move ModelBuilder output into decoder/ subdirectory with standard naming.

    ModelBuilder saves to the Olive cache. The text.json output_dir points to
    a path like 'cpu_and_mobile/models/text.onnx' which contains the ONNX model
    and supporting files. We move them into output_dir/decoder/model.onnx.
    """
    decoder_dir = os.path.join(output_dir, "decoder")
    os.makedirs(decoder_dir, exist_ok=True)

    # Check if text.onnx exists in the Olive output location
    olive_output = os.path.join(output_dir, "text.onnx")
    if os.path.isdir(olive_output):
        for fname in os.listdir(olive_output):
            src = os.path.join(olive_output, fname)
            dst = os.path.join(decoder_dir, fname)
            if os.path.isfile(src):
                os.replace(src, dst)
        shutil.rmtree(olive_output, ignore_errors=True)
        print(f"  [Assemble] Moved decoder files from {olive_output} to {decoder_dir}")

    # Fix external data filename if needed
    _fix_decoder_external_data(decoder_dir)


def export_models(config_dir: str, model_path: str, dtype: str = "f16"):
    """Export all 3 sub-models: text (Olive), vision + embedding (mobius)."""
    output_dir = str(Path(config_dir) / MODELS_DIR)

    print("=== Exporting models ===")

    # Text decoder via Olive/ModelBuilder
    export_text_decoder(config_dir)

    # Assemble decoder into standard directory layout (decoder/model.onnx)
    _assemble_decoder(output_dir)

    # Vision + embedding via mobius
    export_vision_and_embedding(output_dir, model_path, dtype)

    print()


def update_genai_config(output_dir: str = MODELS_DIR, device: str = "cpu"):
    """Patch genai_config.json with embedding/vision sections and processor_config.

    Derives model-specific values from user_script (which lazily loads from HF config)
    to avoid hardcoded constants drifting from the actual checkpoint.
    """
    config_path = Path(output_dir) / "genai_config.json"

    with open(config_path) as f:
        config = json.load(f)

    if device == "gpu":
        provider_options = [
            {
                "cuda": {
                    "enable_cuda_graph": "1",
                    "enable_skip_layer_norm_strict_mode": "1",
                }
            }
        ]
        vision_provider_options = [
            {
                "cuda": {
                    "enable_cuda_graph": "0",
                    "enable_skip_layer_norm_strict_mode": "1",
                }
            }
        ]
    else:
        provider_options = []
        vision_provider_options = []

    session_options = {
        "log_id": "onnxruntime-genai",
        "provider_options": provider_options,
    }
    vision_session_options = {
        "log_id": "onnxruntime-genai",
        "provider_options": vision_provider_options,
    }

    config["model"]["decoder"]["session_options"] = session_options
    config["model"]["decoder"]["filename"] = "decoder/model.onnx"

    # Sync position_ids with what the decoder ONNX model actually supports
    decoder_onnx = Path(output_dir) / "decoder" / "model.onnx"
    if decoder_onnx.exists():
        import onnx

        decoder_model = onnx.load(str(decoder_onnx), load_external_data=False)
        onnx_input_names = {inp.name for inp in decoder_model.graph.input}
        if "position_ids" in onnx_input_names:
            config["model"]["decoder"].setdefault("inputs", {})["position_ids"] = (
                "position_ids"
            )
        else:
            config["model"]["decoder"].get("inputs", {}).pop("position_ids", None)

    config["model"]["embedding"] = {
        "filename": "embedding/model.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": vision_session_options,
    }

    # Derive vision config from user_script constants (sourced from HF config)
    config["model"]["vision"] = {
        "filename": "vision/model.onnx",
        "config_filename": "processor_config.json",
        "spatial_merge_size": user_script.SPATIAL_MERGE_SIZE,
        "patch_size": user_script.PATCH_SIZE,
        "inputs": {"pixel_values": "pixel_values"},
        "outputs": {"image_features": "image_features"},
        "session_options": vision_session_options,
    }

    # Read token IDs from HF config via user_script (lazy-loaded)
    hf_config = user_script._get_config()
    config["model"]["bos_token_id"] = hf_config.text_config.bos_token_id or 1
    config["model"]["context_length"] = 4096
    config["model"]["eos_token_id"] = hf_config.text_config.eos_token_id or 2
    config["model"]["pad_token_id"] = hf_config.text_config.pad_token_id or 11
    config["model"]["image_token_id"] = user_script.IMAGE_TOKEN_ID
    config["model"]["type"] = "mistral3"

    config["search"]["max_length"] = 4096
    config["search"]["top_k"] = 1
    config["search"]["past_present_share_buffer"] = False
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    # Transforms-based processor config (matches ORT GenAI's image preprocessor format)
    processor_config = {
        "processor": {
            "name": "pixtral_image_processor",
            "transforms": [
                {
                    "operation": {
                        "name": "decode_image",
                        "type": "DecodeImage",
                        "attrs": {"color_space": "RGB"},
                    }
                },
                {
                    "operation": {
                        "name": "convert_to_rgb",
                        "type": "ConvertRGB",
                    }
                },
                {
                    "operation": {
                        "name": "resize",
                        "type": "Resize",
                        "attrs": {
                            "height": 1540,
                            "width": 1540,
                            "smart_resize": 1,
                            "min_pixels": 784,
                            "max_pixels": 2371600,
                            "patch_size": user_script.PATCH_SIZE,
                            "merge_size": user_script.SPATIAL_MERGE_SIZE,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "rescale",
                        "type": "Rescale",
                        "attrs": {"rescale_factor": 0.00392156862745098},
                    }
                },
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {
                            "mean": [0.48145466, 0.4578275, 0.40821073],
                            "std": [0.26862954, 0.26130258, 0.27577711],
                        },
                    }
                },
            ],
        }
    }

    processor_path = Path(output_dir) / "processor_config.json"
    with open(processor_path, "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {processor_path}")


def fix_tokenizer(output_dir: str = MODELS_DIR):
    """Fix tokenizer_config.json for onnxruntime-genai compatibility.

    Ministral3's tokenizer uses 'TokenizersBackend' class which isn't supported
    by genai's ort-extensions tokenizer. Change to 'LlamaTokenizer'.
    """
    tc_path = Path(output_dir) / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text(encoding="utf-8"))
        if tc.get("tokenizer_class") == "TokenizersBackend":
            tc["tokenizer_class"] = "LlamaTokenizer"
            tc_path.write_text(
                json.dumps(tc, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("  Fixed tokenizer_class to LlamaTokenizer")


def main():
    parser = argparse.ArgumentParser(description="Optimize Ministral-3-3B ONNX models")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--config-dir", default="cpu_and_mobile")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument(
        "--model-path",
        default=DEFAULT_HF_MODEL,
        help="HuggingFace model ID or local path to dequantized checkpoint",
    )
    parser.add_argument(
        "--dtype",
        default="f16",
        choices=["f16", "f32", "bf16"],
        help="Dtype for mobius vision/embedding export (default: f16)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or str(Path(args.config_dir) / MODELS_DIR)

    if not args.skip_export:
        export_models(args.config_dir, args.model_path, args.dtype)

    print("=== Generating configs ===")
    update_genai_config(output_dir=models_dir, device=args.device)
    fix_tokenizer(output_dir=models_dir)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
