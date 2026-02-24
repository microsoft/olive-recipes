import argparse
import logging
import os

import onnx_ir as ir

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Models larger than this (bytes) use external data format
EXTERNAL_DATA_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2 GB


def _estimate_model_size(model: ir.Model) -> int:
    """Estimate serialized size from initializer shapes and dtypes."""
    total = 0
    for init in model.graph.initializers.values():
        if init.shape is not None and init.dtype is not None:
            numel = 1
            for d in init.shape:
                if isinstance(d, int):
                    numel *= d
                else:
                    numel *= 1  # symbolic dim — underestimate is fine
            bits = ir.DataType(init.dtype).itemsize
            total += numel * bits
    return total


def save_with_external_data(pkg, output_dir: str) -> None:
    """Save package models, using external data for large ones."""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in pkg.items():
        onnx_path = os.path.join(output_dir, f"{name}.onnx")
        est_size = _estimate_model_size(model)
        if est_size > EXTERNAL_DATA_THRESHOLD:
            ext_path = f"{name}.onnx.data"
            print(f"  {name}.onnx  (~{est_size / 1e9:.1f} GB, external data → {ext_path})")
            ir.save(model, onnx_path, external_data=ext_path)
        else:
            print(f"  {name}.onnx  (~{est_size / 1e6:.0f} MB)")
            ir.save(model, onnx_path)


def main():
    parser = argparse.ArgumentParser(description="Export Qwen2.5-VL using onnx-genai-models")
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory (default: models)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="f16",
        choices=["f32", "f16", "bf16"],
        help="Model dtype (default: f16)",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip downloading weights (graph-only export)",
    )
    args = parser.parse_args()

    from onnx_genai_models import build

    print(f"Building {MODEL_ID} ...")
    print(f"  Output:  {args.output}")
    print(f"  Dtype:   {args.dtype}")
    print(f"  Weights: {'no' if args.no_weights else 'yes'}")
    print()

    pkg = build(
        MODEL_ID,
        dtype=args.dtype,
        load_weights=not args.no_weights,
    )

    print(f"\nPackage components: {list(pkg.keys())}")
    save_with_external_data(pkg, args.output)
    print(f"Saved ONNX models to {args.output}/")

    pkg.save_genai_config(args.output)
    pkg.save_tokenizer(args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
