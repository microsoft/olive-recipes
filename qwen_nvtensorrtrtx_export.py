# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Export and validate Qwen3.5/3.6 35B-A3B models for the NvTensorRtRtx EP."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper


def _replace_inputs(node: NodeProto, inputs: list[str]) -> None:
    del node.input[:]
    node.input.extend(inputs)


def _shared_expert_gate_nodes(model: ModelProto) -> list[NodeProto]:
    return [
        node
        for node in model.graph.node
        if node.op_type == "Sigmoid" and node.name.endswith("/shared_expert_gate/Sigmoid")
    ]


def _shared_expert_basename(gate_sigmoid: NodeProto) -> str:
    return gate_sigmoid.name.removesuffix("_gate/Sigmoid")


def _find_down_projection(nodes: list[NodeProto], basename: str) -> NodeProto:
    prefix = f"{basename}/down_proj/MatMul"
    matches = [node for node in nodes if node.op_type == "MatMul" and node.name.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one shared-expert down projection below {basename}, found {len(matches)}")
    return matches[0]


def _is_valid_shared_expert(
    final_mul: NodeProto,
    gate_sigmoid: NodeProto,
    down_projection: NodeProto,
    producer_by_output: dict[str, NodeProto],
) -> bool:
    if down_projection.output[0] not in final_mul.input or gate_sigmoid.output[0] not in final_mul.input:
        return False

    gate_up = producer_by_output.get(down_projection.input[0])
    return gate_up is not None and gate_up.op_type == "Mul" and gate_up is not final_mul


def repair_shared_expert_graph(model: ModelProto) -> int:
    """Repair the duplicate-Mul-name defect in Qwen3.5/3.6 MoE model-builder output.

    Affected ONNX Runtime GenAI exporters use ``<shared_expert>/Mul`` for both
    ``silu(gate) * up`` and ``down * sigmoid(shared_expert_gate)``. Only the
    gate/up multiplication is retained, leaving the down projection ungated.
    This function gives the retained multiplication a unique name and restores
    the missing final gating Mul.
    Already-correct graphs are left unchanged.
    """

    nodes = list(model.graph.node)
    node_by_name = {node.name: node for node in nodes}
    producer_by_output = {output: node for node in nodes for output in node.output}
    repaired = 0

    for gate_sigmoid in _shared_expert_gate_nodes(model):
        basename = _shared_expert_basename(gate_sigmoid)
        final_mul_name = f"{basename}/Mul"
        final_mul = node_by_name.get(final_mul_name)
        if final_mul is None or final_mul.op_type != "Mul":
            raise RuntimeError(f"Missing final shared-expert gating Mul: {final_mul_name}")

        down_projection = _find_down_projection(nodes, basename)
        if _is_valid_shared_expert(final_mul, gate_sigmoid, down_projection, producer_by_output):
            continue

        if not final_mul.output or final_mul.output[0] not in down_projection.input:
            raise RuntimeError(f"Unrecognized shared-expert graph below {basename}")
        if gate_sigmoid.output[0] in final_mul.input or down_projection.output[0] in final_mul.input:
            raise RuntimeError(f"Partially repaired shared-expert graph below {basename}")

        gate_up_name = f"{basename}/gate_up/Mul"
        if gate_up_name in node_by_name:
            raise RuntimeError(f"Duplicate shared-expert gate/up node: {gate_up_name}")

        original_output = final_mul.output[0]
        gate_up_output = f"{gate_up_name}/output_0"
        final_mul.name = gate_up_name
        del final_mul.output[:]
        final_mul.output.append(gate_up_output)
        _replace_inputs(
            down_projection,
            [gate_up_output if name == original_output else name for name in down_projection.input],
        )
        gated_output_mul = helper.make_node(
            "Mul",
            [down_projection.output[0], gate_sigmoid.output[0]],
            [original_output],
            name=final_mul_name,
        )

        consumer_indices = [
            index
            for index, node in enumerate(nodes)
            if node is not down_projection and original_output in node.input
        ]
        if not consumer_indices:
            raise RuntimeError(f"The shared-expert output below {basename} has no consumer")
        nodes.insert(min(consumer_indices), gated_output_mul)
        node_by_name[gate_up_name] = final_mul
        node_by_name[final_mul_name] = gated_output_mul
        producer_by_output[gate_up_output] = final_mul
        producer_by_output[original_output] = gated_output_mul
        repaired += 1

    if repaired:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    return repaired


def _save_graph_only(model: ModelProto, model_path: Path) -> None:
    """Atomically save the graph while preserving existing external weight references."""

    temporary_path = model_path.with_name(f".{model_path.name}.tmp")
    try:
        onnx.save_model(model, temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _iter_values_for_key(value: Any, key: str) -> Iterator[Any]:
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                yield current_value
            yield from _iter_values_for_key(current_value, key)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_values_for_key(item, key)


def _is_enabled(value: Any) -> bool:
    return value is True or value == 1 or (isinstance(value, str) and value.lower() in {"1", "true"})


def validate_genai_config(model_path: Path) -> None:
    config_path = model_path.parent / "genai_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Missing GenAI configuration: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not any(_is_enabled(value) for value in _iter_values_for_key(config, "past_present_share_buffer")):
        raise RuntimeError("The export does not enable the shared past/present buffer")
    if not any(_is_enabled(value) for value in _iter_values_for_key(config, "enable_cuda_graph")):
        raise RuntimeError("The export does not enable CUDA graph for the NvTensorRtRtx EP")


def validate_onnx_graph(model: ModelProto) -> dict[str, int]:
    producer_index = {output: index for index, node in enumerate(model.graph.node) for output in node.output}
    for index, node in enumerate(model.graph.node):
        out_of_order_inputs = [name for name in node.input if producer_index.get(name, -1) >= index]
        if out_of_order_inputs:
            raise RuntimeError(f"Node {node.name} appears before its producers for inputs {out_of_order_inputs}")

    counts = {
        "MatMulNBits": sum(node.op_type == "MatMulNBits" for node in model.graph.node),
        "DequantizeLinear": sum(node.op_type == "DequantizeLinear" for node in model.graph.node),
        "MatMul": sum(node.op_type == "MatMul" for node in model.graph.node),
        "QMoE": sum(node.op_type == "QMoE" for node in model.graph.node),
    }
    if counts["MatMulNBits"]:
        raise RuntimeError(f"Found {counts['MatMulNBits']} MatMulNBits nodes; expected only INT4 QDQ MatMuls")

    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    int4_dq_count = sum(
        node.op_type == "DequantizeLinear"
        and bool(node.input)
        and node.input[0] in initializer_by_name
        and initializer_by_name[node.input[0]].data_type == TensorProto.INT4
        for node in model.graph.node
    )
    if not int4_dq_count:
        raise RuntimeError("The graph does not contain signed INT4 weight DequantizeLinear nodes")
    counts["INT4DequantizeLinear"] = int4_dq_count

    nodes = list(model.graph.node)
    producer_by_output = {output: node for node in nodes for output in node.output}
    gate_sigmoids = _shared_expert_gate_nodes(model)
    if counts["QMoE"] and len(gate_sigmoids) != counts["QMoE"]:
        raise RuntimeError(
            f"Found {counts['QMoE']} QMoE nodes but {len(gate_sigmoids)} shared-expert gate Sigmoid nodes"
        )

    node_by_name = {node.name: node for node in nodes}
    for gate_sigmoid in gate_sigmoids:
        basename = _shared_expert_basename(gate_sigmoid)
        final_mul = node_by_name.get(f"{basename}/Mul")
        down_projection = _find_down_projection(nodes, basename)
        if final_mul is None or not _is_valid_shared_expert(
            final_mul, gate_sigmoid, down_projection, producer_by_output
        ):
            raise RuntimeError(f"Shared-expert gating is incomplete below {basename}")

    counts["SharedExpertGates"] = len(gate_sigmoids)
    return counts


def repair_and_validate(model_path: Path) -> tuple[int, dict[str, int]]:
    model = onnx.load(model_path, load_external_data=False)
    repaired = repair_shared_expert_graph(model)
    counts = validate_onnx_graph(model)
    if repaired:
        _save_graph_only(model, model_path)
    validate_genai_config(model_path)
    return repaired, counts


def _find_model(output_path: Path) -> Path:
    if output_path.is_file() and output_path.suffix == ".onnx":
        return output_path

    direct_model = output_path / "model.onnx"
    if direct_model.is_file():
        return direct_model

    candidates = list(output_path.rglob("model.onnx")) if output_path.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one model.onnx below {output_path}, found {len(candidates)}")
    return candidates[0]


def _default_config(recipe_dir: Path) -> Path:
    configs = list(recipe_dir.glob("*_model_builder_int4.json"))
    if len(configs) != 1:
        raise RuntimeError(f"Expected one ModelBuilder INT4 recipe in {recipe_dir}, found {len(configs)}")
    return configs[0]


def main(recipe_dir: Path | None = None) -> int:
    recipe_dir = (recipe_dir or Path.cwd()).resolve()
    parser = argparse.ArgumentParser(
        description="Export a Qwen model with Olive and verify its NvTensorRtRtx graph and runtime configuration."
    )
    parser.add_argument("--config", type=Path, help="Olive recipe; defaults to the recipe beside export.py")
    parser.add_argument("-o", "--output-path", type=Path, default=Path("output"), help="Export directory")
    parser.add_argument("--log-level", type=int, choices=range(5), help="Olive logging level")
    args = parser.parse_args()

    config_path = (args.config or _default_config(recipe_dir)).resolve()
    output_path = args.output_path.resolve()
    command = [
        sys.executable,
        "-m",
        "olive",
        "run",
        "--config",
        str(config_path),
        "--output_path",
        str(output_path),
    ]
    if args.log_level is not None:
        command.extend(["--log_level", str(args.log_level)])

    subprocess.run(command, cwd=recipe_dir, check=True)
    model_path = _find_model(output_path)
    repaired, counts = repair_and_validate(model_path)
    print(
        f"Validated {model_path}: {counts['INT4DequantizeLinear']} INT4 QDQ weights, "
        f"{counts['MatMul']} MatMuls, {counts['QMoE']} QMoE nodes, "
        f"{counts['SharedExpertGates']} shared-expert gates ({repaired} repaired), "
        "shared past/present buffer enabled, CUDA graph enabled."
    )
    return 0
