"""Package the exported Gemma 4 components with the split NPU vision model."""

import json
import shutil
import tempfile
from pathlib import Path

import onnx


MODELS_DIR = Path(__file__).resolve().parent / "models"
SOURCE = MODELS_DIR.parent.parent / "multi_comp" / "gemma4_onnx"
VISION = MODELS_DIR / "vision_qdq"
OUTPUT = MODELS_DIR / "final"


def rebase_paths(value, old_root, new_root):
    if isinstance(value, dict):
        return {key: rebase_paths(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase_paths(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(old_root):
            return str(new_root / path.relative_to(old_root))
    return value


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing model package: {OUTPUT}")
    for path in (
        SOURCE / "genai_config.json",
        SOURCE / "model_config.json",
        *(SOURCE / name / "model.onnx" for name in ("decoder", "vision_encoder", "audio_encoder", "embedding")),
        VISION / "encoder.onnx",
        VISION / "encoder.onnx.data",
        VISION / "pooler_projector.onnx",
        VISION / "pooler_projector.onnx.data",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")

    encoder = onnx.load(str(VISION / "encoder.onnx"), load_external_data=False).graph
    projector = onnx.load(str(VISION / "pooler_projector.onnx"), load_external_data=False).graph
    if (
        [value.name for value in encoder.input] != ["pixel_values", "pixel_position_ids"]
        or [value.name for value in encoder.output] != ["vision_features"]
        or [value.name for value in projector.input] != ["pixel_position_ids", "vision_features"]
        or [value.name for value in projector.output] != ["image_features"]
        or encoder.output[0].type != projector.input[1].type
        or encoder.input[1].type != projector.input[0].type
    ):
        raise ValueError("Split vision graph inputs, outputs, or boundary types do not match")

    with tempfile.TemporaryDirectory(prefix=".final-", dir=MODELS_DIR) as temporary:
        package = Path(temporary)
        shutil.copytree(SOURCE, package, dirs_exist_ok=True)
        for name in ("encoder", "pooler_projector"):
            model = onnx.load(str(VISION / f"{name}.onnx"), load_external_data=False)
            for tensor in model.graph.initializer:
                for entry in tensor.external_data:
                    if entry.key == "location":
                        if entry.value != f"{name}.onnx.data":
                            raise ValueError(f"Unexpected external weights for {name}: {entry.value}")
                        entry.value = f"model_{name}.onnx.data"
            onnx.save_model(model, str(package / "vision_encoder" / f"model_{name}.onnx"))
            shutil.copy2(
                VISION / f"{name}.onnx.data",
                package / "vision_encoder" / f"model_{name}.onnx.data",
            )

        genai_path = package / "genai_config.json"
        genai = json.loads(genai_path.read_text())
        genai["model"]["vision"]["pipeline"] = {
            "encoder": {
                "filename": "vision_encoder/model_encoder.onnx",
                "inputs": ["pixel_values", "pixel_position_ids"],
                "outputs": ["vision_features"],
            },
            "projector": {
                "filename": "vision_encoder/model_pooler_projector.onnx",
                "inputs": ["vision_features", "pixel_position_ids"],
                "outputs": ["image_features"],
            },
        }
        genai_path.write_text(json.dumps(genai, indent=4) + "\n")

        metadata_path = package / "model_config.json"
        metadata = json.loads(metadata_path.read_text())
        old_root = Path(metadata["config"]["model_path"])
        metadata = rebase_paths(metadata, old_root, OUTPUT)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        package.rename(OUTPUT)
    print(f"Packaged Gemma 4 model at {OUTPUT}")


if __name__ == "__main__":
    main()
