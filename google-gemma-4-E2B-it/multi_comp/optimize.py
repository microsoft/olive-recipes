"""Run the Gemma 4 E2B multi-component optimization pipeline.

Usage:
    python optimize.py --ep OpenVINOExecutionProvider
    python optimize.py --ep QNNExecutionProvider
"""

import argparse
import subprocess
import sys
from pathlib import Path


RECIPE_DIR = Path(__file__).resolve().parent
EP_CONFIGS = {
    "openvino": "ov.json",
    "openvinoexecutionprovider": "ov.json",
    "ov": "ov.json",
    "qnn": "qnn.json",
    "qnnexecutionprovider": "qnn.json",
}


def parse_ep(value: str) -> str:
    ep = value.lower()
    if ep not in EP_CONFIGS:
        raise argparse.ArgumentTypeError(
            "expected one of: OpenVINOExecutionProvider, ov, QNNExecutionProvider, qnn"
        )
    return ep


def run_step(step: int, description: str, args: list[str]) -> None:
    print(f"[{step}/3] {description}", flush=True)
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

    run_step(
        1,
        "Quantizing the Hugging Face components",
        ["run", "--config", "gemma4_quantize.json"],
    )
    run_step(
        2,
        "Exporting the quantized model with Mobius",
        [
            "capture-onnx-graph",
            "--model_name_or_path",
            "gemma4_quantized_hf",
            "--use_mobius_builder",
            "--precision",
            "fp32",
            "--output_path",
            "gemma4_onnx",
        ],
    )
    run_step(3, f"Optimizing for {args.ep}", ["run", "--config", EP_CONFIGS[args.ep]])


if __name__ == "__main__":
    main()
