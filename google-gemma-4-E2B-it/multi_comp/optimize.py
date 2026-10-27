"""Run the Gemma 4 E2B multi-component optimization pipeline.

Usage:
    python optimize.py --ep OpenVINOExecutionProvider
    python optimize.py --ep qnn
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent
EP_ALIASES = {
    "openvino": "ov",
    "openvinoexecutionprovider": "ov",
    "ov": "ov",
    "qnn": "qnn",
    "qnnexecutionprovider": "qnn",
}


def parse_ep(value: str) -> str:
    ep = value.lower()
    if ep not in EP_ALIASES:
        raise argparse.ArgumentTypeError(
            "expected one of: OpenVINOExecutionProvider, ov, QNNExecutionProvider, qnn"
        )
    return EP_ALIASES[ep]


def run_step(step: int, total: int, description: str, args: list[str]) -> None:
    print(f"[{step}/{total}] {description}", flush=True)
    subprocess.run([sys.executable, "-m", "olive", *args], cwd=RECIPE_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize Gemma 4 E2B for OpenVINO or QNN"
    )
    parser.add_argument(
        "--ep",
        required=True,
        type=parse_ep,
        help="Target execution provider: OpenVINOExecutionProvider/ov or QNNExecutionProvider/qnn",
    )
    args = parser.parse_args()

    config_path = RECIPE_DIR / f"{args.ep}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_attributes = config["input_model"]["config"].get("model_attributes")
    accuracy_level = (
        model_attributes.get("matmul_nbits_accuracy_level")
        if model_attributes is not None
        else None
    )
    if accuracy_level is not None and (
        type(accuracy_level) is not int or not 0 <= accuracy_level <= 4
    ):
        parser.error(
            f"{config_path.name}: input_model.config.model_attributes.matmul_nbits_accuracy_level "
            "must be an integer from 0 to 4, or null"
        )
    capture_args = [
        "capture-onnx-graph",
        "--model_name_or_path",
        "gemma4_quantized_hf",
        "--use_mobius_builder",
        "--precision",
        "fp32",
        "--output_path",
        "gemma4_onnx",
    ]
    if accuracy_level is not None:
        capture_args.extend(["--matmul_nbits_accuracy_level", str(accuracy_level)])

    total_steps = 3
    run_step(
        1,
        total_steps,
        "Quantizing the Hugging Face components",
        ["run", "--config", "gemma4_quantize.json"],
    )
    run_step(
        2,
        total_steps,
        "Exporting the quantized model with Mobius",
        capture_args,
    )

    run_step(
        total_steps,
        total_steps,
        "Optimizing for OpenVINO" if args.ep == "ov" else "Optimizing for QNN",
        ["run", "--config", f"{args.ep}.json"],
    )


if __name__ == "__main__":
    main()
