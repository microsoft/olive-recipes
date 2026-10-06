"""Run the Gemma 4 E2B multi-component optimization pipeline.

Usage:
    python optimize.py --ep OpenVINOExecutionProvider
    python optimize.py --ep QNNExecutionProvider
"""

import argparse
import json
import subprocess
import sys
import tempfile
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


def prepare_qnn_config(output_path: Path) -> None:
    import onnx

    config = json.loads((RECIPE_DIR / "qnn.json").read_text(encoding="utf-8"))
    model_path = RECIPE_DIR / config["input_model"]["config"]["model_path"] / "decoder" / "model.onnx"
    model = onnx.load(model_path, load_external_data=False)
    lm_head_nodes = [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMulNBits" and node.name.startswith("decoder/lm_head/")
    ]
    softcap_nodes = [
        node.name
        for node in model.graph.node
        if node.name.startswith(("decoder/Div_node_", "decoder/Tanh_node_", "decoder/Mul_node_"))
    ]
    if len(lm_head_nodes) != 1 or len(softcap_nodes) != 3:
        raise ValueError(
            f"Expected one quantized lm_head and three logits softcap nodes in {model_path}; "
            f"found {len(lm_head_nodes)} and {len(softcap_nodes)}."
        )

    config["passes"]["matmul_nbits_to_qdq_decoder"]["nodes_to_exclude"] = lm_head_nodes
    config["passes"]["sq_decoder"]["nodes_to_exclude"] = softcap_nodes
    output_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")


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
    with tempfile.TemporaryDirectory() as directory:
        config_path = EP_CONFIGS[args.ep]
        if config_path == "qnn.json":
            resolved_config = Path(directory) / config_path
            prepare_qnn_config(resolved_config)
            config_path = str(resolved_config)
        run_step(3, f"Optimizing for {args.ep}", ["run", "--config", config_path])


if __name__ == "__main__":
    main()
