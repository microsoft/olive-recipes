"""End-to-end optimization pipeline for Ministral-3-3B ONNX models.

Uses Olive MobiusBuilder pass for vision and embedding export (reliable
dynamo-free ONNX construction), and Olive/ModelBuilder for text decoder
export (GQA + INT4).

Pipeline:
    1. Text decoder: Olive/ModelBuilder (k_quant_mixed INT4)
    2. Vision + embedding: Olive/MobiusBuilder (FP16 for cuda/webgpu, FP32 for cpu_and_mobile, via vision_embedding_export.json)
    3. Vision quantization: Olive (INT8 RTN, per vision.json)

Architecture difference from Qwen VLM recipes:
    Qwen uses Olive passes for all 3 sub-models (export + optimization).
    Ministral uses MobiusBuilder for vision/embedding because Pixtral's dynamic
    image dimensions cause torch.onnx.export/dynamo failures.  MobiusBuilder
    produces already-optimized graphs (fused MHA, SkipLayerNorm, FP16).

Usage:
    python optimize.py --config-dir cuda --device gpu
    python optimize.py --config-dir cpu_and_mobile --device cpu
    python optimize.py --config-dir webgpu --device webgpu
    python optimize.py --config-dir cuda --device gpu --skip-export
    python optimize.py --config-dir cpu_and_mobile --device cpu --model-path /local/dequantized/checkpoint
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODELS_DIR = "models"
MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"

# Lazy-loaded HuggingFace config (avoids import-time network access)
_HF_CONFIG = None


def _get_hf_config():
    """Load and cache the HuggingFace model config.

    Always loads from MODEL_NAME (the canonical HF model ID) rather than
    --model-path, because the config values (image_token_id, patch_size, etc.)
    are architecture constants that don't change between checkpoints.
    """
    global _HF_CONFIG
    if _HF_CONFIG is None:
        from transformers import Mistral3Config

        _HF_CONFIG = Mistral3Config.from_pretrained(MODEL_NAME)
    return _HF_CONFIG


def export_text_decoder(config_dir: str, models_dir: str):
    """Export text decoder using Olive/ModelBuilder (GQA + quantization).

    Loads text.json as a dict and overrides output_dir to write directly
    to <models_dir>/decoder. ModelBuilder also generates genai_config.json,
    tokenizer, and chat_template inside decoder/ — we move them to the
    models root where the VLM pipeline expects them.
    """
    try:
        from olive import run
    except ImportError:
        from olive.workflows import run

    config_path = Path(config_dir) / "text.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Text config not found: {config_path}")

    # Load config as dict and override output_dir to write directly to models_dir
    with open(config_path) as f:
        config = json.load(f)
    config["output_dir"] = os.path.join(models_dir, "decoder")

    print(f"  [Olive] Exporting text decoder from {config_path}...")
    run(config)

    # Move shared configs from decoder/ to models root for VLM pipeline
    decoder_dir = Path(models_dir) / "decoder"
    for filename in (
        "genai_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        src = decoder_dir / filename
        if src.exists():
            shutil.move(str(src), str(Path(models_dir) / filename))


def export_vision_and_embedding(config_dir: str, models_dir: str, model_path: str = MODEL_NAME):
    """Export vision encoder and embedding using Olive MobiusBuilder pass.

    Runs <config_dir>/vision_embedding_export.json which calls MobiusBuilder with
    components_to_export=["vision_encoder", "embedding"], writing two
    sub-directories under models_dir:
        vision_encoder/model.onnx  — exported vision encoder, fed into INT8 quantization step
        embedding/model.onnx       — exported embedding (FP16/FP32, not quantized)

    Mobius constructs the ONNX graph declaratively and applies pretrained
    weights, avoiding torch.onnx.export dynamo issues with Pixtral's
    dynamic image dimensions. Precision is controlled by the
    vision_embedding_export.json config (fp16 for cuda/webgpu, fp32 for cpu_and_mobile).
    """
    try:
        from olive import run
    except ImportError:
        from olive.workflows import run

    config_path = Path(config_dir) / "vision_embedding_export.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Vision/embedding export config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Olive writes output model to output_dir. Override to models_dir so it lands
    # alongside decoder/ in the same parent directory.
    config["output_dir"] = models_dir
    if model_path != MODEL_NAME:
        config["input_model"]["model_path"] = model_path

    print(f"  [Olive] Exporting vision encoder and embedding from {config_path}...")
    run(config)

    # Olive's CompositeModelHandler writes flat ONNX files to output_dir:
    #   models_dir/vision_encoder.onnx  + models_dir/vision_encoder.onnx.data
    #   models_dir/embedding.onnx       + models_dir/embedding.onnx.data
    #
    # Reorganize into the subdirectory layout expected by quantize_vision_and_embedding:
    #   models_dir/vision_encoder/model.onnx  + {component}.onnx.data
    #   models_dir/embedding/model.onnx       + {component}.onnx.data
    #
    # The data file keeps its original name (e.g. vision_encoder.onnx.data) so that
    # the relative external_data reference baked into model.onnx remains valid.
    for component in ("vision_encoder", "embedding"):
        src_onnx = Path(models_dir) / f"{component}.onnx"
        if src_onnx.exists():
            dst_dir = Path(models_dir) / component
            dst_dir.mkdir(exist_ok=True)
            shutil.move(str(src_onnx), str(dst_dir / "model.onnx"))
            src_data = Path(models_dir) / f"{component}.onnx.data"
            if src_data.exists():
                # Keep original filename — model.onnx references it by this relative path
                shutil.move(str(src_data), str(dst_dir / f"{component}.onnx.data"))


def quantize_vision_and_embedding(config_dir: str, models_dir: str):
    """Apply quantization to Olive-exported vision and embedding models.

    Loads vision.json as a dict and overrides model_path and output_dir.
    Vision encoder is sourced from vision_encoder/ (MobiusBuilder output)
    and quantized output is written to vision/ (the name ort-genai expects).

    Embedding stays FP16 (cuda/webgpu) or FP32 (cpu_and_mobile) — no quantization needed, no embedding.json.
    """
    try:
        from olive import run
    except ImportError:
        from olive.workflows import run

    for component in ("vision", "embedding"):
        config_path = Path(config_dir) / f"{component}.json"
        if not config_path.exists():
            continue

        # MobiusBuilder outputs vision_encoder/ but ort-genai expects vision/.
        # Source from vision_encoder/ and write quantized output to vision/.
        source_dir = "vision_encoder" if component == "vision" else component
        component_onnx = os.path.join(models_dir, source_dir, "model.onnx")
        if not os.path.exists(component_onnx):
            print(
                f"  [WARN] {component_onnx} not found, skipping {component} quantization"
            )
            continue

        # Load config as dict and override paths to target models_dir directly
        with open(config_path) as f:
            config = json.load(f)
        config["input_model"]["model_path"] = component_onnx
        config["output_dir"] = os.path.join(models_dir, component)

        print(f"  [Olive] Quantizing {component} from {config_path}...")
        run(config)

        # Olive catches pass failures internally and returns without raising.
        # Guard _strip_unused_initializers so a silent quantization failure
        # doesn't propagate as a confusing FileNotFoundError.
        output_onnx = os.path.join(models_dir, component, "model.onnx")
        if not os.path.exists(output_onnx):
            print(f"  [WARN] Olive produced no output for {component} — quantization failed")
            continue
        _strip_unused_initializers(output_onnx)

    # Clean up intermediate vision_encoder/ only if vision quantization succeeded.
    # If quantization failed, preserve the intermediate for debugging.
    vision_dir = os.path.join(models_dir, "vision")
    vision_encoder_dir = os.path.join(models_dir, "vision_encoder")
    if os.path.exists(os.path.join(vision_dir, "model.onnx")) and os.path.exists(vision_encoder_dir):
        print("  Cleaning up intermediate vision_encoder export...")
        shutil.rmtree(vision_encoder_dir, ignore_errors=True)


def _strip_unused_initializers(onnx_path: str):
    """Remove unused initializers and re-save to shrink the external data file.

    Olive's OnnxBlockWiseRtnQuantization keeps original weights alongside
    the new quantized weights. Stripping the unused originals typically
    reduces the data file by ~87% (e.g., 1.7GB → 220MB for the vision model).
    """
    if not os.path.exists(onnx_path):
        return

    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    model = onnx.load(onnx_path)

    used_names = set()
    for node in model.graph.node:
        for inp in node.input:
            used_names.add(inp)
    for inp in model.graph.input:
        used_names.add(inp.name)

    before = len(model.graph.initializer)
    new_inits = [init for init in model.graph.initializer if init.name in used_names]
    removed = before - len(new_inits)

    if removed == 0:
        return

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)

    for init in model.graph.initializer:
        del init.external_data[:]

    data_name = "model.onnx.data"
    data_path = os.path.join(os.path.dirname(onnx_path), data_name)
    if os.path.exists(data_path):
        os.remove(data_path)

    convert_model_to_external_data(
        model, all_tensors_to_one_file=True, location=data_name, size_threshold=1024
    )
    onnx.save(model, onnx_path)

    data_mb = os.path.getsize(data_path) / 1e6 if os.path.exists(data_path) else 0
    print(f"  [Cleanup] Stripped {removed} unused initializers → {data_mb:.0f} MB")


def export_models(
    config_dir: str, model_path: str, dtype: str = "f16", models_dir: str | None = None
):
    """Export all 3 sub-models: text (Olive/ModelBuilder), vision + embedding (Olive/MobiusBuilder).

    All outputs go directly to models_dir:
        decoder/           — INT4 k_quant text decoder (from text.json / ModelBuilder)
        vision_encoder/    — FP16/FP32 vision encoder (from MobiusBuilder, input for INT8 quant)
        embedding/         — FP16/FP32 embedding (from MobiusBuilder, not quantized)
        vision/            — INT8 quantized vision (from vision.json)

    Note: precision for vision/embedding export is set in vision_embedding_export.json
    (fp16 for cuda/webgpu, fp32 for cpu_and_mobile). The --dtype CLI arg is accepted for
    backward compatibility but does not control export precision — precision is set in the
    JSON config files.
    """
    if dtype != "f16":
        import warnings

        warnings.warn(
            "--dtype is deprecated and has no effect. Export precision is controlled by "
            "vision_embedding_export.json in the config directory.",
            DeprecationWarning,
            stacklevel=2,
        )
    if models_dir is None:
        models_dir = str(Path(config_dir) / MODELS_DIR)

    print("=== Exporting models ===")

    # Text decoder via Olive/ModelBuilder (GQA + INT4 k_quant)
    export_text_decoder(config_dir, models_dir)

    # Vision encoder + embedding via Olive/MobiusBuilder (FP16 for cuda/webgpu, FP32 for cpu_and_mobile)
    export_vision_and_embedding(config_dir, models_dir, model_path)

    # INT8 quantization of vision encoder (embedding stays FP16/FP32)
    quantize_vision_and_embedding(config_dir, models_dir)

    print()


def update_genai_config(output_dir: str = MODELS_DIR, device: str = "cpu"):
    """Patch genai_config.json with embedding/vision sections and processor_config.

    Derives model-specific values from the HuggingFace config (lazily loaded)
    to avoid hardcoded constants drifting from the actual checkpoint.
    """
    config_path = Path(output_dir) / "genai_config.json"

    with open(config_path) as f:
        config = json.load(f)

    if device == "gpu":
        # CUDA graph capture is unsupported for VLMs with dynamic image sizes.
        # Disable for all models (matches Qwen VLM recipe convention).
        provider_options = [
            {
                "cuda": {
                    "enable_cuda_graph": "0",
                    "enable_skip_layer_norm_strict_mode": "1",
                }
            }
        ]
        vision_provider_options = provider_options
    elif device == "webgpu":
        provider_options = [{"webgpu": {}}]
        vision_provider_options = provider_options
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

    # Vision config — values derived from HF config to stay in sync with checkpoint
    hf_config = _get_hf_config()
    config["model"]["vision"] = {
        "filename": "vision/model.onnx",
        "config_filename": "processor_config.json",
        "spatial_merge_size": hf_config.spatial_merge_size,
        "patch_size": hf_config.vision_config.patch_size,
        "inputs": {"pixel_values": "pixel_values"},
        "outputs": {"image_features": "image_features"},
        "session_options": vision_session_options,
    }

    # Add VLM-specific fields not generated by ModelBuilder.
    # Don't override context_length or max_length — PR #2077's ModelBuilder
    # sets these correctly (context_length=262144, max_length=32768).
    config["model"]["image_token_id"] = hf_config.image_token_index

    # Override search defaults for VLM: greedy decoding, no KV sharing
    config["search"]["top_k"] = 1
    config["search"]["past_present_share_buffer"] = False

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
                            "patch_size": hf_config.vision_config.patch_size,
                            "merge_size": hf_config.spatial_merge_size,
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
                {
                    "operation": {
                        "name": "permute",
                        "type": "Permute3D",
                        "attrs": {"dims": [2, 0, 1]},
                    }
                },
                {
                    "operation": {
                        "name": "pixtral_image_sizes",
                        "type": "PixtralImageSizes",
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
    parser.add_argument("--device", choices=["gpu", "cpu", "webgpu"], default="cpu")
    parser.add_argument("--config-dir", default="cpu_and_mobile")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument(
        "--model-path",
        default=MODEL_NAME,
        help="HuggingFace model ID or local path to dequantized checkpoint",
    )
    parser.add_argument(
        "--dtype",
        default="f16",
        choices=["f16", "f32", "bf16"],
        help="Quantization precision for the text decoder (INT4 via Olive/ModelBuilder). "
        "Does not affect vision/embedding export — precision is set in "
        "vision_embedding_export.json (FP16 for cuda/webgpu, FP32 for cpu_and_mobile). (default: f16)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or str(Path(args.config_dir) / MODELS_DIR)

    if not args.skip_export:
        export_models(args.config_dir, args.model_path, args.dtype, models_dir)

    print("=== Generating configs ===")
    update_genai_config(output_dir=models_dir, device=args.device)
    fix_tokenizer(output_dir=models_dir)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
