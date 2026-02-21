"""End-to-end optimization pipeline for Qwen3-VL ONNX models.

Usage:
    # Full pipeline: export + optimize + INT4 quantize
    python optimize.py --device cpu --quantize

    # Export only (no quantization)
    python optimize.py --device cpu

    # Quantize only (skip export, models already exist)
    python optimize.py --quantize --skip-export
"""
import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper, numpy_helper

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODELS_DIR = "models"


# =============================================================================
# 1. Olive Export
# =============================================================================

def export_models():
    """Run Olive export for all 3 sub-models (embedding, text, vision)."""
    from olive import run

    print("=== Exporting models with Olive ===")
    for config in ("embedding.json", "text.json", "vision.json"):
        print(f"  Running {config}...")
        run(config)
    print()


# =============================================================================
# 2. Config Generation
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
# 3. Graph Optimization (com.microsoft opset fixup + Cast chain elimination)
# =============================================================================

def optimize_graph(model_path: str):
    """Apply com.microsoft opset fixup and ORT Cast chain elimination."""
    import onnxruntime as ort

    print(f"  Optimizing graph: {os.path.basename(model_path)}")

    # Ensure com.microsoft opset is registered in all ONNX functions.
    # GraphSurgeries may add com.microsoft ops without updating function opsets.
    model = onnx.load(model_path)
    _ensure_com_microsoft_opset(model)
    onnx.save(model, model_path)
    del model

    # Re-run ORT basic optimization with Cast chain elimination explicitly enabled
    # (disabled by default via kOrtSessionOptionsEnableCastChainElimination).
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = model_path
    sess_options.add_session_config_entry("session.enable_cast_chain_elimination", "1")
    ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])


# =============================================================================
# 4. Quantization
# =============================================================================

def _gemm_to_matmul(model: onnx.ModelProto) -> int:
    """Convert Gemm nodes to MatMul+Add for INT4 quantization compatibility."""
    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}
    nodes_to_remove = []
    nodes_to_add = []

    for node in graph.node:
        if node.op_type != "Gemm":
            continue

        alpha = beta = 1.0
        transA = transB = 0
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = attr.f
            elif attr.name == "beta":
                beta = attr.f
            elif attr.name == "transA":
                transA = attr.i
            elif attr.name == "transB":
                transB = attr.i

        if alpha != 1.0 or beta != 1.0 or transA != 0:
            continue

        A, B = node.input[0], node.input[1]
        C = node.input[2] if len(node.input) > 2 else None
        Y = node.output[0]

        if transB:
            if B in initializer_map:
                init = initializer_map[B]
                w_t = numpy_helper.to_array(init).T.copy()
                new_init = numpy_helper.from_array(w_t, name=B)
                for i, existing in enumerate(graph.initializer):
                    if existing.name == B:
                        graph.initializer[i].CopyFrom(new_init)
                        break
                matmul_B = B
            else:
                transpose_out = f"{node.name}_transpose_B"
                nodes_to_add.append(helper.make_node(
                    "Transpose", [B], [transpose_out], name=f"{node.name}_Transpose", perm=[1, 0]))
                matmul_B = transpose_out
        else:
            matmul_B = B

        if C:
            matmul_out = f"{node.name}_matmul_out"
            nodes_to_add.append(helper.make_node("MatMul", [A, matmul_B], [matmul_out], name=f"{node.name}_MatMul"))
            nodes_to_add.append(helper.make_node("Add", [matmul_out, C], [Y], name=f"{node.name}_Add"))
        else:
            nodes_to_add.append(helper.make_node("MatMul", [A, matmul_B], [Y], name=f"{node.name}_MatMul"))

        nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.node.remove(node)
    graph.node.extend(nodes_to_add)
    return len(nodes_to_remove)


def _ensure_com_microsoft_opset(model: onnx.ModelProto):
    """Ensure com.microsoft opset is declared at model and function level."""
    existing = {op.domain for op in model.opset_import}
    if "com.microsoft" not in existing:
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    for func in model.functions:
        func_domains = {op.domain for op in func.opset_import}
        if "com.microsoft" not in func_domains:
            func.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def quantize_int4(model_path: str, block_size: int = 128, include_gather: bool = False):
    """Apply INT4 block-wise RTN quantization (MatMulNBits)."""
    from onnxruntime.quantization import matmul_nbits_quantizer

    model = onnx.load(model_path)
    converted = _gemm_to_matmul(model)
    if converted:
        print(f"    Converted {converted} Gemm -> MatMul+Add")

    temp_path = model_path + ".temp"
    onnx.save(model, temp_path)
    del model

    op_types = ("MatMul", "Gather") if include_gather else None
    model = onnx.load(temp_path)
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model=model, block_size=block_size, is_symmetric=True,
        accuracy_level=4, op_types_to_quantize=op_types,
    )
    quantizer.process()
    result = quantizer.model.model
    _ensure_com_microsoft_opset(result)
    onnx.save(result, model_path)
    os.remove(temp_path)


def quantize_model(model_path: str, block_size: int = 128, include_gather: bool = False):
    """Quantize a single model to INT4 (MatMulNBits)."""
    orig_size = os.path.getsize(model_path) / 1024 / 1024
    quantize_int4(model_path, block_size=block_size, include_gather=include_gather)
    new_size = os.path.getsize(model_path) / 1024 / 1024
    model = onnx.load(model_path, load_external_data=False)
    ops = Counter(n.op_type for n in model.graph.node)
    key_ops = {k: v for k, v in ops.items() if k in ("MatMulNBits", "MatMul")}
    print(f"    {orig_size:.0f} MB -> {new_size:.0f} MB  {dict(key_ops)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimize Qwen3-VL ONNX models")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu",
                        help="Target device (default: cpu)")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize vision/embedding to INT4 (MatMulNBits)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip Olive export (models already exist)")
    parser.add_argument("--block-size", type=int, default=128,
                        help="INT4 block size (default: 128)")
    parser.add_argument("--models-dir", default=MODELS_DIR,
                        help="Models directory (default: models)")
    args = parser.parse_args()

    # Step 1: Export
    if not args.skip_export:
        export_models()

    # Step 2: Generate configs
    print("=== Generating configs ===")
    update_genai_config(output_dir=args.models_dir, device=args.device)
    print()

    # Step 3: Graph optimization (vision + embedding)
    print("=== Optimizing graphs ===")
    for name in ("vision.onnx", "embedding.onnx"):
        path = os.path.join(args.models_dir, name)
        if os.path.exists(path):
            optimize_graph(path)
    print()

    # Step 4: Quantize
    if args.quantize:
        print("=== Quantizing to INT4 ===")

        vision_path = os.path.join(args.models_dir, "vision.onnx")
        if os.path.exists(vision_path):
            print("  Vision:")
            quantize_model(vision_path, block_size=args.block_size)

        emb_path = os.path.join(args.models_dir, "embedding.onnx")
        if os.path.exists(emb_path):
            print("  Embedding:")
            quantize_model(emb_path, block_size=args.block_size, include_gather=True)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
