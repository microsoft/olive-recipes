# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------
"""End-to-end optimization pipeline for Qwen3.6-35B-A3B MoE VLM.

Exports three sub-models (vision encoder, text embedding, text decoder),
applies graph optimizations and INT4 quantization via Olive passes.

Usage:
    python optimize.py --config-dir cuda --device gpu
    python optimize.py --config-dir cuda --device gpu --skip-export
    python optimize.py --config-dir cpu_and_mobile --device cpu
"""
import argparse
import json
import logging
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODELS_DIR = "models"


def _read_model_name(config_dir: str) -> str:
    """Read the HuggingFace model name from the Olive text config."""
    text_cfg = _read_text_config(config_dir)
    return text_cfg["input_model"]["model_path"]


def _read_text_config(config_dir: str) -> dict:
    """Read the Olive config for the text decoder."""
    return json.loads((Path(config_dir) / "text.json").read_text())


def _needs_cuda_int4_packing(config_dir: str) -> bool:
    """Return whether the text decoder export needs CUDA INT4 weight pre-packing."""
    text_cfg = _read_text_config(config_dir)
    target = text_cfg.get("target")
    systems = text_cfg.get("systems", {})

    target_systems = [systems[target]] if target in systems else systems.values()
    uses_cuda = any(
        accelerator.get("device") == "gpu" or "CUDAExecutionProvider" in accelerator.get("execution_providers", [])
        for system_cfg in target_systems
        for accelerator in system_cfg.get("accelerators", [])
    )

    for pass_cfg in text_cfg.get("passes", {}).values():
        if pass_cfg.get("precision") != "int4":
            continue
        if pass_cfg.get("int4_algo_config") == "rtn_last" or uses_cuda:
            return True
    return False


def check_int4_cuda_support():
    """Ensure the environment can pre-pack INT4 weights for the CUDA QMoE kernel.

    INT4 export uses the ONNX Runtime CUDA mixed-GEMM weight pre-packing op, which
    is only available in onnxruntime-gpu and requires a CUDA-capable GPU. Fail fast
    with an actionable message instead of crashing deep inside the export.
    """
    import torch

    try:
        from onnxruntime.capi import _pybind_state as _pybind
    except ImportError:
        _pybind = None

    if not (_pybind and hasattr(_pybind, "pack_weights_for_cuda_mixed_gemm") and torch.cuda.is_available()):
        raise RuntimeError(
            "INT4 export requires CUDA weight pre-packing. Please install onnxruntime-gpu "
            "(with the 'pack_weights_for_cuda_mixed_gemm' op) and run on a CUDA-capable GPU machine."
        )


def export_models(config_dir: str):
    """Run Olive for all 3 sub-models (text, embedding, vision)."""
    from olive import run

    config_path = Path(config_dir)
    print(f"=== Running Olive pipelines (configs from {config_path}) ===")
    for config in ("text.json", "embedding.json", "vision.json"):
        print(f"  Running {config}...")
        run(str(config_path / config))
    print()


def update_genai_config(
    output_dir: str,
    model_name: str,
    device: str = "cpu",
    context_length: int = 4096,
    text_only: bool = False,
    gpu_mem_limit_gb: float | None = None,
    disable_cuda_graph: bool = False,
):
    """Patch genai_config.json with embedding/vision sections and processor_config.

    Reads token IDs, vision parameters, and preprocessor settings from the
    HuggingFace model config rather than hardcoding them.

    ``text_only`` omits the vision encoder section (and its image inputs) to save
    GPU memory (~3.3 GB for the fp32 vision encoder) for text-only deployments
    such as a 24 GB RTX 4090. ``gpu_mem_limit_gb`` sets a CUDA arena ceiling (a
    safety cap, not a memory saver) and ``disable_cuda_graph`` turns off decoder
    CUDA graph (on by default for decode throughput).
    """
    from transformers import AutoConfig, GenerationConfig
    from huggingface_hub import hf_hub_download

    hf_config = AutoConfig.from_pretrained(model_name)
    gen_config = GenerationConfig.from_pretrained(model_name)
    vc = hf_config.vision_config

    config_path = Path(output_dir) / "genai_config.json"
    with open(config_path) as f:
        config = json.load(f)

    if device == "gpu":
        vision_provider_options = [
            {"cuda": {"enable_cuda_graph": "0"}}
        ]
    else:
        vision_provider_options = []

    vision_session_options = {"log_id": "onnxruntime-genai", "provider_options": vision_provider_options}

    config["model"]["decoder"]["filename"] = "text.onnx"

    # Tune the decoder's CUDA arena to reduce GPU memory. ModelBuilder emits the
    # decoder session_options; patch its CUDA provider in place so we keep its
    # other options while bounding arena growth. arena_extend_strategy=1
    # (kSameAsRequested) avoids the default power-of-two (0) over-allocation
    # spike during prefill (this is the real memory saver). onnxruntime-genai
    # consumes the arena strategy as an integer (0=kNextPowerOfTwo,
    # 1=kSameAsRequested), not the ORT enum-name string.
    #
    # CUDA graph is enabled on the decoder: during token-by-token decode the
    # shapes are static (1 token; the hybrid recurrent/conv states and the small
    # GQA KV cache use fixed buffers), so graph capture/replay removes per-kernel
    # launch overhead for a meaningful decode-throughput gain at negligible
    # memory cost. Vision/embedding keep cuda_graph off (variable shapes).
    #
    # gpu_mem_limit is a HARD ceiling, not a memory saver: it does not reduce
    # usage and will OOM if the model needs more. Only set it (via
    # gpu_mem_limit_gb) as a safety cap >= the resident weight footprint
    # (e.g. ~23 GB on a 24 GB card); leave unset to rely on the arena strategy.
    #
    # enable_skip_layer_norm_strict_mode is intentionally removed: ORT >= 1.27
    # computes SkipLayerNorm in fp32 internally, so strict mode is unnecessary.
    # Re-add it ("1") only when targeting ORT <= 1.26.
    if device == "gpu":
        dec_so = config["model"]["decoder"].setdefault("session_options", {})
        dec_po = dec_so.setdefault("provider_options", [{"cuda": {}}])
        for entry in dec_po:
            cuda_opts = entry.get("cuda")
            if cuda_opts is None:
                continue
            cuda_opts.pop("enable_skip_layer_norm_strict_mode", None)
            cuda_opts["enable_cuda_graph"] = "0" if disable_cuda_graph else "1"
            cuda_opts["arena_extend_strategy"] = "1"  # 1 = kSameAsRequested
            if gpu_mem_limit_gb is not None:
                cuda_opts["gpu_mem_limit"] = str(int(gpu_mem_limit_gb * 1024 ** 3))

    embedding_inputs = {"input_ids": "input_ids"}
    if not text_only:
        embedding_inputs["image_features"] = "image_features"
    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": embedding_inputs,
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": vision_session_options,
    }

    # The vision encoder (fp32, ~3.3 GB) is only needed for image inputs. Omit it
    # for text-only deployments to fit a 24 GB GPU.
    config["model"].pop("vision", None)
    if not text_only:
        config["model"]["vision"] = {
            "filename": "vision.onnx",
            "config_filename": "processor_config.json",
            "spatial_merge_size": vc.spatial_merge_size,
            "tokens_per_second": 2.0,
            "patch_size": vc.patch_size,
            "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
            "outputs": {"image_features": "image_features"},
            "session_options": vision_session_options,
        }

    config["model"]["bos_token_id"] = gen_config.bos_token_id
    config["model"]["context_length"] = context_length
    config["model"]["eos_token_id"] = gen_config.eos_token_id
    config["model"]["pad_token_id"] = gen_config.pad_token_id
    config["model"]["image_token_id"] = hf_config.image_token_id
    config["model"]["video_token_id"] = hf_config.video_token_id
    config["model"]["vision_start_token_id"] = hf_config.vision_start_token_id

    config["search"]["max_length"] = context_length
    if gen_config.top_k is not None:
        config["search"]["top_k"] = gen_config.top_k
    if gen_config.top_p is not None and config["search"].get("top_p") is None:
        config["search"]["top_p"] = gen_config.top_p

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    pp_path = hf_hub_download(model_name, "preprocessor_config.json")
    with open(pp_path) as f:
        pp = json.load(f)
    pp_size = pp.get("size", {})

    processor_config = {
        "processor": {
            "name": "qwen2_5_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {"operation": {"name": "resize", "type": "Resize", "attrs": {
                    "width": 960, "height": 672, "smart_resize": 1,
                    "min_pixels": pp_size.get("shortest_edge", 65536),
                    "max_pixels": pp_size.get("longest_edge", 16777216),
                    "patch_size": vc.patch_size,
                    "merge_size": vc.spatial_merge_size,
                }}},
                {"operation": {"name": "rescale", "type": "Rescale", "attrs": {
                    "rescale_factor": 1.0 / 255,
                }}},
                {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
                    "mean": pp.get("image_mean", [0.5, 0.5, 0.5]),
                    "std": pp.get("image_std", [0.5, 0.5, 0.5]),
                    "qwen2_5_vl": 1,
                }}},
                {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
                    "patch_size": vc.patch_size,
                    "temporal_patch_size": vc.temporal_patch_size,
                    "merge_size": vc.spatial_merge_size,
                }}},
            ],
        }
    }

    processor_path = Path(output_dir) / "processor_config.json"
    with open(processor_path, "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {processor_path}")


def fix_tokenizer(output_dir: str = MODELS_DIR):
    """Fix tokenizer.json for C++ std::regex compatibility."""
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
    parser = argparse.ArgumentParser(description="Optimize Qwen3.6-35B-A3B MoE VLM")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--config-dir", default="cuda")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--text-only", action="store_true",
        help="Omit the vision encoder section to save ~3.3 GB GPU memory (text-only deployments, e.g. 24 GB RTX 4090).",
    )
    parser.add_argument(
        "--gpu-mem-limit-gb", type=float, default=None,
        help="CUDA arena ceiling (GB) — a SAFETY CAP, not a memory saver. Set >= resident weight footprint (e.g. 23 on a 24 GB card); leave unset to rely on kSameAsRequested.",
    )
    parser.add_argument(
        "--disable-cuda-graph", action="store_true",
        help="Disable decoder CUDA graph (on by default; it boosts decode throughput at negligible memory cost).",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or str(Path(args.config_dir) / MODELS_DIR)
    model_name = _read_model_name(args.config_dir)

    if not args.skip_export:
        if _needs_cuda_int4_packing(args.config_dir):
            check_int4_cuda_support()
        export_models(args.config_dir)

    print("=== Generating configs ===")
    update_genai_config(
        output_dir=models_dir,
        model_name=model_name,
        device=args.device,
        context_length=args.context_length,
        text_only=args.text_only,
        gpu_mem_limit_gb=args.gpu_mem_limit_gb,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    fix_tokenizer(output_dir=models_dir)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
