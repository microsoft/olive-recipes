"""Fold complete BF16 -> FP32 norm -> BF16 chains for existing ORT CUDA kernels."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _attribute(node, name, default=None):
    return next((helper.get_attribute_value(a) for a in node.attribute if a.name == name), default)


def _cast(node, to):
    return (
        node.op_type == "Cast"
        and node.domain in ("", "ai.onnx")
        and len(node.input) == len(node.output) == 1
        and _attribute(node, "to") == to
    )


def _scale_tensor(tensor, base_dir, name):
    # Loading a copy keeps external-data metadata in the source model untouched.
    copy = TensorProto()
    copy.CopyFrom(tensor)
    values = numpy_helper.to_array(copy, base_dir=str(base_dir)).astype("<f4", copy=False)
    bits = values.view("<u4")
    if not np.isfinite(values).all() or np.any(bits & 0xFFFF):
        return None
    return TensorProto(
        name=name,
        data_type=TensorProto.BFLOAT16,
        dims=list(tensor.dims),
        raw_data=(bits >> 16).astype("<u2").tobytes(),
    )


def rewrite(source, output, *, expected_casts=None, excluded_initializers=()):
    """Write a separate graph beside its original external weights; return counts.

    Only complete, single-consumer chains are rewritten. Mixed input/output norm
    types are deliberately excluded so no new ORT kernel registrations are needed.
    """
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.parent.resolve() != source.parent:
        raise ValueError("Output must be a new filename beside the source model and external weights")
    model = onnx.load(source, load_external_data=False)
    graph = model.graph
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for node in graph.node for a in node.attribute):
        raise ValueError("Subgraphs and their implicit captures are not supported")
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in [*graph.input, *graph.output, *graph.value_info]
        if value.type.HasField("tensor_type")
    }
    weights = {tensor.name: tensor for tensor in graph.initializer}
    types.update({name: tensor.data_type for name, tensor in weights.items()})
    producers, consumers = {}, defaultdict(list)
    names = set(types) | set(weights)
    for i, node in enumerate(graph.node):
        names.update(node.input)
        names.update(node.output)
        for name in node.output:
            if name in producers:
                raise ValueError(f"Multiple producers for {name}")
            producers[name] = i
        for name in node.input:
            consumers[name].append(i)
        if node.op_type == "Cast" and node.domain in ("", "ai.onnx") and len(node.output) == 1:
            types.setdefault(node.output[0], _attribute(node, "to"))
    graph_inputs = {value.name for value in graph.input}
    graph_outputs = {value.name for value in graph.output}
    excluded = set(excluded_initializers)
    removed_nodes, removed_values = set(), set()
    converted, skipped = {}, Counter()
    rewritten = 0
    for node in graph.node:
        if node.op_type != "SimplifiedLayerNormalization" or node.domain not in ("", "ai.onnx"):
            continue
        if (len(node.input) != 2 or len(node.output) != 1
                or _attribute(node, "stash_type", TensorProto.FLOAT) != TensorProto.FLOAT):
            skipped["unsupported_norm_signature"] += 1
            continue
        before_index = producers.get(node.input[0])
        after_indices = consumers[node.output[0]]
        if before_index is None or len(after_indices) != 1:
            skipped["not_a_complete_chain"] += 1
            continue
        before, after = graph.node[before_index], graph.node[after_indices[0]]
        if not (_cast(before, TensorProto.FLOAT) and _cast(after, TensorProto.BFLOAT16)):
            skipped["not_a_complete_chain"] += 1
            continue
        if (types.get(before.input[0]) != TensorProto.BFLOAT16
                or types.get(node.input[0]) != TensorProto.FLOAT
                or types.get(node.output[0]) != TensorProto.FLOAT
                or types.get(after.output[0]) != TensorProto.BFLOAT16):
            skipped["unsupported_or_unknown_types"] += 1
            continue
        if (len(consumers[before.output[0]]) != 1
                or before.output[0] in graph_outputs or node.output[0] in graph_outputs):
            skipped["shared_or_exposed_intermediate"] += 1
            continue
        scale_name = node.input[1]
        scale = weights.get(scale_name)
        if (scale is None or scale.data_type != TensorProto.FLOAT
                or scale_name in graph_inputs or scale_name in excluded):
            skipped["nonconstant_or_excluded_scale"] += 1
            continue
        if scale_name not in converted:
            name = scale_name + "_norm_cast_bf16"
            while name in names:
                name += "_"
            tensor = _scale_tensor(scale, source.parent, name)
            if tensor is None:
                skipped["inexact_or_nonfinite_scale"] += 1
                continue
            names.add(name)
            converted[scale_name] = tensor
        removed_nodes.update((before_index, after_indices[0]))
        removed_values.update((before.output[0], node.output[0]))
        node.input[0] = before.input[0]
        node.input[1] = converted[scale_name].name
        node.output[0] = after.output[0]
        rewritten += 1
    if expected_casts is not None and len(removed_nodes) != expected_casts:
        raise ValueError(f"Expected {expected_casts} removable Casts, found {len(removed_nodes)}")
    if not rewritten:
        raise ValueError(f"No eligible complete BF16 norm/Cast chains found: {dict(skipped)}")
    retained = [node for i, node in enumerate(graph.node) if i not in removed_nodes]
    del graph.node[:]
    graph.node.extend(retained)
    retained = [value for value in graph.value_info if value.name not in removed_values]
    del graph.value_info[:]
    graph.value_info.extend(retained)
    used = {name for node in graph.node for name in node.input} | graph_inputs | graph_outputs
    retained = [tensor for tensor in graph.initializer if tensor.name not in converted or tensor.name in used]
    del graph.initializer[:]
    graph.initializer.extend(retained + list(converted.values()))
    # Serialize only the graph: never load or rewrite the large external weight files.
    data = model.SerializeToString()
    with output.open("xb") as stream:
        stream.write(data)
    return {
        "norms_rewritten": rewritten,
        "casts_removed": len(removed_nodes),
        "scales_converted": len(converted),
        "skipped": dict(skipped),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-casts", type=int)
    parser.add_argument("--exclude-initializer", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(rewrite(args.source, args.output, expected_casts=args.expected_casts,
                             excluded_initializers=args.exclude_initializer), indent=2))
