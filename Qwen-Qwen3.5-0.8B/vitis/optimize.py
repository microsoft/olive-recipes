"""Run component builds and retain the unchanged Mobius runtime assets."""

import json
import shutil
from pathlib import Path

from olive import run


def main():
    with Path("config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    source = Path(config["input_model"]["config"]["model_path"])
    destination = Path(config["builds"]["_default"]["output_dir"])
    for name in ("decoder", "vision_encoder", "embedding"):
        if not (source / name / "model.onnx").is_file():
            raise FileNotFoundError(f"Missing {source / name / 'model.onnx'}. Run the Mobius export first.")
    if not (source / "genai_config.json").is_file():
        raise FileNotFoundError(f"Missing {source / 'genai_config.json'}. Export the full VLM, not selected components.")
    if destination.exists():
        raise FileExistsError(f"{destination} already exists. Use a fresh output directory to avoid mixing artifacts.")

    run(config)

    for name in ("decoder", "vision_encoder"):
        if not (destination / name / "model.onnx").is_file():
            raise FileNotFoundError(f"The {name} build did not produce {destination / name / 'model.onnx'}.")
    shutil.copytree(source / "embedding", destination / "embedding", dirs_exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "model_config.json":
            shutil.copy2(path, destination / path.name)
    print(f"Optimized components: {destination}")


if __name__ == "__main__":
    main()
