# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""End-to-end optimization pipeline for VideoChat-Flash ONNX models.

Orchestrates three Olive export+optimise runs (embedding, text, vision),
then writes the GenAI runtime configuration files.

Produces 3 ONNX models:
  vision.onnx     — InternVideo2-1B + ToMe16 projector, fp32
  embedding.onnx  — embed_tokens + visual feature merge, fp32
  text.onnx       — Qwen2.5-7B decoder, int4 (quantized via ModelBuilder)

Usage:
    python optimize.py --config-dir cpu_and_mobile --device cpu
    python optimize.py --config-dir cpu_and_mobile --device cpu --mode video
    python optimize.py --config-dir cpu_and_mobile --device cpu --skip-export
    python optimize.py --config-dir cpu_and_mobile --device cpu --staging-dir D:/staging
"""
import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODELS_DIR = "models"

MODEL_ID = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "added_tokens.json", "special_tokens_map.json",
    "chat_template.jinja",
]


# ======================================================================
# 0.  Pre-flight checks
# ======================================================================

def check_prerequisites():
    """Warn about potential issues before starting the pipeline."""
    try:
        from huggingface_hub import HfApi
        token = HfApi().token
        if not token:
            print("WARNING: HuggingFace token not set.")
            print("  If the Olive cache is empty, the text export will fail.")
            print("  Run:  huggingface-cli login")
            print()
    except Exception:
        pass


# ======================================================================
# 1.  Clean + Olive Export
# ======================================================================

def clean_output_dirs(models_dir: Path, staging_dir: Path | None = None):
    """Remove stale model files before a fresh export.

    Without this, leftover files from a previous run can collide with
    new Olive outputs (subdirectory names vs. existing file names) and
    cause the normalize step to pair protobufs with wrong data files.
    """
    if staging_dir:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
            print(f"  Cleaned staging dir: {staging_dir}")
        staging_dir.mkdir(parents=True, exist_ok=True)
    else:
        for name in ("embedding.onnx", "text.onnx", "vision.onnx"):
            target = models_dir / name
            # Remove subdirectory created by Olive
            if target.is_dir():
                shutil.rmtree(target)
            # Remove flat files from previous run
            for f in models_dir.glob(f"{name}*"):
                if f.is_file():
                    f.unlink()
        models_dir.mkdir(parents=True, exist_ok=True)
    print("  Output directories ready")
    print()


def export_models(config_dir: str, models_dir: Path, mode: str = "image",
                  staging_dir: Path | None = None):
    """Run Olive pipelines for all three sub-models.

    Each config runs in a separate subprocess so that memory is fully
    released between exports (the full HF model is ~15 GB on its own).

    The JSON configs' output_dir is always overridden via a per-run
    temp config so that Olive writes where the rest of the pipeline
    (normalize, assemble, config-update) expects:
      - When staging_dir is provided, Olive writes per-component
        subdirectories under staging_dir.
      - Otherwise Olive writes flat to models_dir/{component}.onnx,
        matching the layout the JSONs assume by default.
    """
    import subprocess
    import sys

    vision_config = "vision_image.json" if mode == "image" else "vision.json"
    configs = [
        ("embedding", "embedding.json"),
        ("text", "text.json"),
        ("vision", vision_config),
    ]

    config_path = Path(config_dir)
    print(f"=== Running Olive pipelines (configs from {config_path}, mode={mode}) ===")
    if staging_dir:
        print(f"    Staging directory: {staging_dir}")
    else:
        print(f"    Models directory: {models_dir}")

    for component, config_file in configs:
        cfg_file = config_path / config_file

        with open(cfg_file) as f:
            cfg = json.load(f)
        if staging_dir:
            cfg["output_dir"] = str(staging_dir / component)
        else:
            cfg["output_dir"] = str(models_dir / f"{component}.onnx")

        fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix=f"{component}_")
        tmp = Path(tmp_path)
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=4)
        run_config = str(tmp)

        print(f"\n--- {config_file} ---", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "olive", "run", "--config", run_config],
            cwd=str(Path(__file__).parent),
        )

        tmp.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Olive failed for {config_file} (exit code {result.returncode})"
            )
    print()


# ======================================================================
# 2.  Normalize Olive outputs — rename model files + fix data refs
# ======================================================================

def normalize_model_output(output_dir: Path, component: str):
    """Ensure a model has the correct protobuf and data file names.

    Handles two Olive output layouts:
      a) Subdirectory: output_dir/{component}.onnx/ containing model.onnx
         (when output_dir is e.g. cpu_and_mobile/models and Olive creates
         a {component}.onnx subdirectory)
      b) Flat directory: output_dir/ containing model.onnx
         (when output_dir is e.g. staging/embedding)

    In both cases, ensures the final result is:
      output_dir/{component}.onnx  (protobuf)
      output_dir/{component}.onnx.data  (external weights)
    """
    import onnx

    target_proto = output_dir / f"{component}.onnx"
    target_data = f"{component}.onnx.data"

    # Case (a): Olive created a subdirectory named {component}.onnx
    subdir = output_dir / f"{component}.onnx"
    if subdir.is_dir():
        # Move all files out of the subdirectory to its parent.
        # A file may share the subdir's name (e.g. text.onnx/text.onnx);
        # use a temp name to break the collision.
        tmp_name = None
        for f in list(subdir.iterdir()):
            dest = output_dir / f.name
            if dest == subdir:
                tmp_name = f.name
                f.replace(subdir / f"_tmp_{f.name}")
            else:
                f.replace(dest)
        if tmp_name:
            tmp_file = subdir / f"_tmp_{tmp_name}"
            tmp_file.replace(output_dir / f"_tmp_{tmp_name}")
            subdir.rmdir()
            (output_dir / f"_tmp_{tmp_name}").replace(output_dir / tmp_name)
        else:
            subdir.rmdir()
        print(f"  Flattened {component}.onnx/ subdirectory")

    # Rename model.onnx -> {component}.onnx
    src = output_dir / "model.onnx"
    if src.exists() and not target_proto.exists():
        src.replace(target_proto)
        src_data = output_dir / "model.onnx.data"
        if src_data.exists():
            src_data.replace(output_dir / target_data)
        print(f"  Renamed model.onnx -> {component}.onnx")

    if not target_proto.exists():
        print(f"  WARNING: {target_proto} not found, skipping")
        return

    # Fix internal external data references if mismatched
    proto = onnx.load(str(target_proto), load_external_data=False)
    old_data_names = set()
    for tensor in proto.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                old_data_names.add(entry.value)

    if not old_data_names:
        print(f"  {component}.onnx OK (no external data)")
        return

    if old_data_names == {target_data} and (output_dir / target_data).exists():
        print(f"  {component}.onnx OK (data refs correct)")
        return

    changed = False
    for old_name in old_data_names:
        if old_name == target_data:
            continue
        old_file = output_dir / old_name
        if old_file.exists():
            old_file.replace(output_dir / target_data)
            print(f"  Renamed data {old_name} -> {target_data}")
            changed = True

    if not (output_dir / target_data).exists():
        print(f"  WARNING: data file {target_data} not found in {output_dir}")

    for tensor in proto.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value != target_data:
                entry.value = target_data
                changed = True

    if changed:
        with open(target_proto, "wb") as f:
            f.write(proto.SerializeToString())
        print(f"  Fixed {component}.onnx data refs -> {target_data}")


def normalize_all_models(models_dir: Path, staging_dir: Path | None = None):
    """Normalize all model outputs after Olive export."""
    print("=== Normalizing model outputs ===")
    for component in ("embedding", "vision", "text"):
        if staging_dir:
            output_dir = staging_dir / component
        else:
            output_dir = models_dir
        if not output_dir.exists():
            print(f"  WARNING: {output_dir} not found, skipping {component}")
            continue
        normalize_model_output(output_dir, component)
    print()


# ======================================================================
# 3.  Assemble final models directory (staging mode only)
# ======================================================================

def assemble_from_staging(staging_dir: Path, models_dir: Path):
    """Copy final ONNX models + tokenizer from staging to the models dir.

    Only used when --staging-dir is provided.  Cleans stale files from
    the target directory first.
    """
    print(f"=== Assembling models from {staging_dir} -> {models_dir} ===")

    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.is_file():
                f.unlink()

    models_dir.mkdir(parents=True, exist_ok=True)

    for component in ("text", "embedding", "vision"):
        staging = staging_dir / component
        if not staging.exists():
            continue
        for f in staging.glob(f"{component}.onnx*"):
            size_gb = f.stat().st_size / (1024 ** 3)
            label = f"({size_gb:.2f} GB)" if size_gb > 0.01 else ""
            print(f"  Copying {f.name} {label}")
            shutil.copy2(f, models_dir / f.name)

    text_staging = staging_dir / "text"
    copied = 0
    for fname in TOKENIZER_FILES:
        src = text_staging / fname
        if src.exists():
            shutil.copy2(src, models_dir / fname)
            copied += 1
    print(f"  Copied {copied} tokenizer files")

    genai_src = text_staging / "genai_config.json"
    if genai_src.exists():
        shutil.copy2(genai_src, models_dir / "genai_config.json")
        print(f"  Copied genai_config.json")
    else:
        raise FileNotFoundError(
            f"genai_config.json not found in {text_staging}. "
            "Re-run without --skip-export."
        )

    print()


# ======================================================================
# 4.  GenAI Runtime Config + Processor Config
# ======================================================================

def update_genai_config(output_dir: str = MODELS_DIR, device: str = "cpu"):
    """Patch genai_config.json with embedding / vision sections."""
    config_path = Path(output_dir) / "genai_config.json"

    with open(config_path) as f:
        config = json.load(f)

    if device == "gpu":
        provider_options = [
            {"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1"}}
        ]
    else:
        provider_options = []

    session_options = {
        "log_id": "onnxruntime-genai",
        "provider_options": provider_options,
    }

    config["model"]["type"] = "qwen3_vl"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is not None:
        config["model"]["pad_token_id"] = tokenizer.pad_token_id
    else:
        config["model"]["pad_token_id"] = config["model"]["bos_token_id"]

    for stale_key in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        config["model"].pop(stale_key, None)

    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {
            "input_ids": "input_ids",
            "image_features": "image_features",
        },
        "outputs": {
            "inputs_embeds": "inputs_embeds",
        },
        "session_options": session_options,
    }

    config["model"]["vision"] = {
        "filename": "vision.onnx",
        "config_filename": "processor_config.json",
        "num_visual_tokens": 64,
        "inputs": {
            "pixel_values": "images",
        },
        "outputs": {
            "image_features": "visual_tokens",
        },
        "session_options": session_options,
    }

    if config["search"].get("top_k") is None:
        config["search"]["top_k"] = 50
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    processor_config = {
        "processor": {
            "name": "internvideo2_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage",
                               "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {"operation": {"name": "resize", "type": "Resize",
                               "attrs": {"width": 224, "height": 224}}},
                {"operation": {"name": "rescale", "type": "Rescale",
                               "attrs": {"rescale_factor": 0.00392156862745098}}},
                {"operation": {"name": "normalize", "type": "Normalize",
                               "attrs": {"mean": [0.485, 0.456, 0.406],
                                         "std": [0.229, 0.224, 0.225]}}},
            ],
        }
    }

    proc_path = Path(output_dir) / "processor_config.json"
    with open(proc_path, "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {proc_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimize VideoChat-Flash ONNX models"
    )
    parser.add_argument(
        "--device", choices=["gpu", "cpu"], default="cpu",
        help="Target device (default: cpu)",
    )
    parser.add_argument(
        "--config-dir", default="cpu_and_mobile",
        help="Directory containing Olive JSON configs (default: cpu_and_mobile)",
    )
    parser.add_argument(
        "--mode", choices=["image", "video"], default="image",
        help="Vision export mode (default: image)",
    )
    parser.add_argument(
        "--staging-dir", type=str, default=None,
        help="Stage intermediate files on a separate drive",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip Olive export (models already exist)",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Models output directory (default: <config-dir>/models)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else Path(args.config_dir) / MODELS_DIR
    staging_dir = Path(args.staging_dir) if args.staging_dir else None

    check_prerequisites()

    if not args.skip_export:
        print("=== Preparing output directories ===")
        clean_output_dirs(models_dir, staging_dir=staging_dir)
        export_models(args.config_dir, models_dir, mode=args.mode, staging_dir=staging_dir)

    normalize_all_models(models_dir, staging_dir=staging_dir)

    if staging_dir:
        assemble_from_staging(staging_dir, models_dir)

    print("=== Generating configs ===")
    update_genai_config(output_dir=str(models_dir), device=args.device)

    print("\nDone. Models ready at:", models_dir)


if __name__ == "__main__":
    main()
