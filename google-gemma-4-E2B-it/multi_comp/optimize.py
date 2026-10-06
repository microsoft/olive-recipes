"""Run the Gemma 4 E2B multi-component optimization pipeline.

Usage:
    python optimize.py --ep OpenVINOExecutionProvider
    python optimize.py --ep QNNExecutionProvider
"""

import argparse
from copy import deepcopy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


RECIPE_DIR = Path(__file__).resolve().parent
QNN_VISION_OUTPUT = RECIPE_DIR / "gemma4_qnn_vision"
QNN_DECODER_OUTPUT = RECIPE_DIR / "gemma4_qnn_decoder"
QNN_OUTPUT = RECIPE_DIR / "gemma4_qnn"
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


def prepare_qnn_decoder_config(output_path: Path) -> None:
    import onnx
    from olive.model import CompositeModelHandler

    config = json.loads((RECIPE_DIR / "qnn_decoder.json").read_text(encoding="utf-8"))
    model_dir = RECIPE_DIR / config["input_model"]["config"]["model_path"]
    composite_model = CompositeModelHandler(model_path=str(model_dir))
    decoder_model = dict(composite_model.get_model_components())["decoder"]
    decoder_model.model_attributes = {
        **(decoder_model.model_attributes or {}),
        "additional_files": [str(model_dir / "genai_config.json")],
    }
    config["input_model"] = composite_model.to_json()
    model_path = decoder_model.model_path
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


def load_composite_config(package_dir: Path) -> dict:
    config_path = package_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Olive package config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config.get("config") or {}
    names = model_config.get("model_component_names") or []
    components = model_config.get("model_components") or []
    if config.get("type", "").lower() != "compositemodel" or len(names) != len(components):
        raise ValueError(f"Invalid CompositeModel package config: {config_path}")
    return config


def relative_config_path(path_value: str, root_value: str) -> Path | None:
    value = str(path_value).replace("\\", "/").rstrip("/")
    root = str(root_value).replace("\\", "/").rstrip("/")
    if value.casefold() == root.casefold():
        return Path()
    prefix = f"{root}/"
    if not value.casefold().startswith(prefix.casefold()):
        return None

    parts = PurePosixPath(value[len(prefix) :]).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid package-relative path: {path_value}")
    return Path(*parts)


def rebase_config_paths(value, old_root: str, new_root: Path):
    if isinstance(value, dict):
        return {key: rebase_config_paths(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase_config_paths(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        relative = relative_config_path(value, old_root)
        if relative is not None:
            return str(new_root / relative)
    return value


def get_component_map(config: dict) -> dict[str, dict]:
    model_config = config["config"]
    return dict(zip(model_config["model_component_names"], model_config["model_components"]))


def get_component_dir(package_dir: Path, config: dict, component_name: str) -> Path:
    component = get_component_map(config)[component_name]
    relative = relative_config_path(
        component["config"]["model_path"],
        config["config"]["model_path"],
    )
    if relative is None or relative == Path():
        raise ValueError(f"Component {component_name!r} is not stored in its own package directory.")
    component_dir = package_dir / relative
    if not component_dir.is_dir():
        raise FileNotFoundError(f"Missing component directory: {component_dir}")
    return relative


def merge_qnn_outputs() -> None:
    for package_dir in (QNN_VISION_OUTPUT, QNN_DECODER_OUTPUT):
        if not package_dir.is_dir():
            raise FileNotFoundError(f"Missing QNN component package: {package_dir}")
    if QNN_OUTPUT.exists():
        raise FileExistsError(f"Final QNN output already exists; use a clean output directory: {QNN_OUTPUT}")

    vision_config = load_composite_config(QNN_VISION_OUTPUT)
    decoder_config = load_composite_config(QNN_DECODER_OUTPUT)
    vision_components = get_component_map(vision_config)
    decoder_components = get_component_map(decoder_config)
    selected_components = ("embedding", "vision_encoder")
    missing = [
        name
        for name in selected_components
        if name not in vision_components or name not in decoder_components
    ]
    if missing:
        raise ValueError(f"Missing QNN merge component(s): {missing}")

    output_root = QNN_OUTPUT.resolve()
    merged_config = rebase_config_paths(
        deepcopy(decoder_config),
        decoder_config["config"]["model_path"],
        output_root,
    )
    merged_components = get_component_map(merged_config)
    for name in selected_components:
        merged_components[name].clear()
        merged_components[name].update(
            rebase_config_paths(
                deepcopy(vision_components[name]),
                vision_config["config"]["model_path"],
                output_root,
            )
        )

    decoder_attributes = merged_config["config"].setdefault("model_attributes", {})
    optimized_components = set(decoder_attributes.get("assembled_components") or [])
    optimized_components.update(vision_config["config"].get("model_attributes", {}).get("assembled_components") or [])
    decoder_attributes["assembled_components"] = [
        name for name in merged_config["config"]["model_component_names"] if name in optimized_components
    ]

    with tempfile.TemporaryDirectory(prefix=".gemma4_qnn_merge_", dir=RECIPE_DIR) as directory:
        staging_dir = Path(directory) / QNN_OUTPUT.name
        shutil.copytree(
            QNN_DECODER_OUTPUT,
            staging_dir,
            ignore=shutil.ignore_patterns(".builds"),
        )
        for name in selected_components:
            vision_relative = get_component_dir(QNN_VISION_OUTPUT, vision_config, name)
            decoder_relative = get_component_dir(QNN_DECODER_OUTPUT, decoder_config, name)
            if vision_relative != decoder_relative:
                raise ValueError(
                    f"Component {name!r} uses different package paths: "
                    f"{vision_relative} and {decoder_relative}"
                )
            destination = staging_dir / decoder_relative
            shutil.rmtree(destination)
            shutil.copytree(QNN_VISION_OUTPUT / vision_relative, destination)

        (staging_dir / "model_config.json").write_text(
            json.dumps(merged_config, indent=4) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(QNN_OUTPUT)

    print(f"Merged QNN package: {QNN_OUTPUT}", flush=True)


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

    total_steps = 5 if args.ep == "qnn" else 3
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

    if args.ep == "qnn":
        run_step(
            3,
            total_steps,
            "Optimizing vision and embedding for QNN",
            ["run", "--config", "qnn_vision.json"],
        )
        with tempfile.TemporaryDirectory() as directory:
            resolved_config = Path(directory) / "qnn_decoder.json"
            prepare_qnn_decoder_config(resolved_config)
            run_step(
                4,
                total_steps,
                "Optimizing the decoder for QNN",
                ["run", "--config", str(resolved_config)],
            )
        print("[5/5] Merging the QNN component packages", flush=True)
        merge_qnn_outputs()
        return

    run_step(
        3,
        total_steps,
        "Optimizing for OpenVINO",
        ["run", "--config", "ov.json"],
    )


if __name__ == "__main__":
    main()
