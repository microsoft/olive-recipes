"""Optimize the three Qwen3-VL ONNX components and assemble a runtime package."""

import shutil
from pathlib import Path

from olive import run


BASE_PACKAGE = Path("exported_vlm_pkg")
OUTPUT_PACKAGE = Path("optimized_vlm_pkg")
CONFIG = Path("vlm_optimize_components.json")


def main() -> None:
    if not (BASE_PACKAGE / "genai_config.json").is_file():
        raise FileNotFoundError(
            "exported_vlm_pkg is missing. Export the FP32 Mobius package first."
        )
    if OUTPUT_PACKAGE.exists():
        shutil.rmtree(OUTPUT_PACKAGE)

    run(str(CONFIG))

    for path in BASE_PACKAGE.iterdir():
        if path.is_file() and path.name != "model_config.json":
            shutil.copy2(path, OUTPUT_PACKAGE / path.name)

    print(f"Optimized ORT GenAI package: {OUTPUT_PACKAGE}")


if __name__ == "__main__":
    main()
