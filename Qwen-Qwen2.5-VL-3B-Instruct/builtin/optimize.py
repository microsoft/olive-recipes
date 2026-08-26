"""Export and optimize Qwen2.5-VL with Olive and Mobius.

Each target directory contains one Olive multi-build config. Mobius first
exports the complete three-component package once; the config then applies
the target-specific pipeline to decoder, vision_encoder, and embedding.
"""

import argparse
import json
import shutil
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
COMPONENTS = ("decoder", "vision_encoder", "embedding")


def _system_config(device: str) -> tuple[str, dict]:
    if device == "gpu":
        name = "local_gpu"
        accelerator = {
            "device": "gpu",
            "execution_providers": ["CUDAExecutionProvider"],
        }
    else:
        name = "local_cpu"
        accelerator = {
            "device": "cpu",
            "execution_providers": ["CPUExecutionProvider"],
        }
    return name, {
        "type": "LocalSystem",
        "accelerators": [accelerator],
    }


def export_with_mobius(config_dir: Path, device: str, output_dir: Path) -> None:
    """Export one complete Mobius package through Olive."""
    from olive import run

    system_name, system = _system_config(device)
    precision = "fp16" if device == "gpu" else "fp32"
    export_config = {
        "input_model": {
            "type": "HfModel",
            "config": {
                "model_path": MODEL_ID,
                "task": "image-text-to-text",
                "load_kwargs": {"trust_remote_code": True},
            },
        },
        "systems": {system_name: system},
        "passes": {
            "mobius": {
                "type": "MobiusBuilder",
                "precision": precision,
            }
        },
        "engine": {
            "host": system_name,
            "target": system_name,
            "evaluate_input_model": False,
            "cache_dir": str(config_dir / "mobius_cache"),
            "output_dir": str(output_dir),
        },
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    print(f"=== Exporting all components with Olive + Mobius ({precision}) ===")
    run(export_config)


def run_component_builds(config_path: Path, base_dir: Path, models_dir: Path) -> None:
    """Run all three component builds from one Olive config."""
    from olive import run

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    config["input_model"]["config"]["model_path"] = str(base_dir)
    config["builds"]["_default"]["output_dir"] = str(models_dir)

    if models_dir.exists():
        shutil.rmtree(models_dir)
    print(
        f"=== Running decoder, vision_encoder, and embedding builds from {config_path} ==="
    )
    run(config)


def copy_runtime_artifacts(base_dir: Path, models_dir: Path) -> None:
    """Copy Mobius-generated GenAI, tokenizer, and processor metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for path in base_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, models_dir / path.name)


def _validate_base_package(base_dir: Path) -> None:
    missing = [
        str(base_dir / component / "model.onnx")
        for component in COMPONENTS
        if not (base_dir / component / "model.onnx").is_file()
    ]
    missing.extend(
        str(base_dir / name)
        for name in ("genai_config.json", "processor_config.json", "tokenizer.json")
        if not (base_dir / name).is_file()
    )
    if missing:
        raise FileNotFoundError(
            "Mobius base package is incomplete; missing " + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize Qwen2.5-VL with Olive and Mobius"
    )
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument(
        "--config-dir",
        default="cpu_and_mobile",
        help="Target directory containing config.json",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse <config-dir>/mobius_base and rerun the component builds",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Final package directory (default: <config-dir>/models)",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    config_path = config_dir / "config.json"
    base_dir = config_dir / "mobius_base"
    models_dir = Path(args.models_dir) if args.models_dir else config_dir / "models"

    expected_device = "gpu" if config_dir.name == "cuda" else "cpu"
    if args.device != expected_device:
        raise ValueError(
            f"{config_dir} targets {expected_device}; got --device {args.device}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Olive multi-build config not found: {config_path}")
    if not args.skip_export:
        export_with_mobius(config_dir, args.device, base_dir)
    _validate_base_package(base_dir)
    run_component_builds(config_path, base_dir, models_dir)
    copy_runtime_artifacts(base_dir, models_dir)

    print(f"Done. ORT GenAI package: {models_dir}")


if __name__ == "__main__":
    main()
