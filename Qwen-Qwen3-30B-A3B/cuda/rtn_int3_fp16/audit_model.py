from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PureWindowsPath

import numpy as np
import onnx


EXPERT_INPUTS = {
    2: ("fc1_experts_weights", (128, 1536, 768), onnx.TensorProto.UINT8),
    3: ("fc1_scales", (128, 1536, 32), onnx.TensorProto.FLOAT16),
    5: ("fc2_experts_weights", (128, 2048, 288), onnx.TensorProto.UINT8),
    6: ("fc2_scales", (128, 2048, 12), onnx.TensorProto.FLOAT16),
}
ABSENT_INPUTS = {
    4: "fc1_bias",
    7: "fc2_bias",
    8: "fc3_weights",
    9: "fc3_scales",
    10: "fc3_bias",
    11: "fc1_zero_points",
    12: "fc2_zero_points",
    13: "fc3_zero_points",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(root: Path, location: str) -> Path:
    relative = Path(location)
    require(
        bool(location)
        and not relative.is_absolute()
        and not PureWindowsPath(location).drive
        and "\\" not in location
        and ".." not in relative.parts,
        f"Unsafe file location: {location!r}",
    )
    resolved = (root / relative).resolve()
    require(
        resolved.is_relative_to(root.resolve()),
        f"File escapes model root: {location!r}",
    )
    require(resolved.is_file(), f"Missing file: {location!r}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_tensors(message):
    if isinstance(message, onnx.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            for child in value if field.is_repeated else (value,):
                yield from iter_tensors(child)


def external_ranges(model, root: Path):
    ranges = []
    for tensor in iter_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        require(
            len(metadata) == len(tensor.external_data),
            f"Duplicate external keys: {tensor.name}",
        )
        location = metadata.get("location", "")
        path = safe_file(root, location)
        require(
            all(dim >= 0 for dim in tensor.dims), f"Negative dimensions: {tensor.name}"
        )
        dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
        packed_bits = {"INT4": 4, "UINT4": 4, "FLOAT4E2M1": 4, "INT2": 2, "UINT2": 2}
        bits = packed_bits.get(dtype)
        if bits is None:
            bits = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize * 8
        expected = (math.prod(tensor.dims) * bits + 7) // 8
        offset = int(metadata.get("offset", "0"))
        length = int(metadata.get("length", str(expected)))
        require(
            offset >= 0
            and length == expected
            and offset + length <= path.stat().st_size,
            f"Invalid external range: {tensor.name}: offset={offset}, length={length}, expected={expected}",
        )
        ranges.append(
            {
                "tensor": tensor.name,
                "location": location,
                "offset": offset,
                "length": length,
            }
        )
    return ranges


def attributes(node):
    result = {}
    for attr in node.attribute:
        require(
            attr.name not in result, f"Duplicate attribute: {node.name}: {attr.name}"
        )
        value = onnx.helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        require(
            isinstance(value, (str, int, float, list)),
            f"Unsupported attribute: {attr.name}",
        )
        result[attr.name] = value
    return result


class StaticExpertResolver:
    def __init__(self, model, tensors):
        self.tensors = tensors
        self.producers = {}
        for node in model.graph.node:
            for output in node.output:
                if output:
                    require(
                        output not in self.producers and output not in tensors,
                        f"Duplicate tensor producer: {output}",
                    )
                    self.producers[output] = node
        self.inputs = {value.name for value in model.graph.input}
        self.metadata = {}
        for value in (*model.graph.value_info, *model.graph.input, *model.graph.output):
            self.metadata.setdefault(value.name, []).append(value.type.tensor_type)
        self.active = set()

    def check_metadata(self, name, shape, dtype):
        for info in self.metadata.get(name, []):
            require(
                not info.elem_type or info.elem_type == dtype,
                f"Metadata dtype mismatch: {name}",
            )
            if info.HasField("shape"):
                require(
                    len(info.shape.dim) == len(shape), f"Metadata rank mismatch: {name}"
                )
                require(
                    all(
                        not dim.HasField("dim_value") or dim.dim_value == actual
                        for dim, actual in zip(info.shape.dim, shape)
                    ),
                    f"Metadata shape mismatch: {name}",
                )

    def node(self, name, allowed):
        require(name not in self.inputs, f"Runtime-dependent expert input: {name}")
        node = self.producers.get(name)
        require(node is not None, f"Missing initializer or static producer: {name}")
        require(
            node.domain in ("", "ai.onnx") and node.op_type in allowed,
            f"Unsupported static transformation: {name}: {node.op_type}",
        )
        require(len(node.output) == 1, f"Invalid static outputs: {name}")
        return node

    def constant(self, name, source, trace):
        key = ("constant", name)
        require(key not in self.active, f"Cyclic static shape: {name}")
        self.active.add(key)
        try:
            require(name not in self.inputs, f"Runtime-dependent reshape shape: {name}")
            if name in self.tensors:
                tensor = self.tensors[name]
                require(
                    tensor.data_type == onnx.TensorProto.INT64
                    and tensor.data_location != onnx.TensorProto.EXTERNAL
                    and math.prod(tensor.dims) <= 64,
                    f"Expected small inline INT64 shape constant: {name}",
                )
                result = onnx.numpy_helper.to_array(tensor)
            else:
                node = self.node(name, {"Constant", "Shape", "Gather", "Concat"})
                if node.op_type == "Constant":
                    require(
                        not node.input and len(node.attribute) == 1,
                        f"Invalid shape Constant: {name}",
                    )
                    attr = node.attribute[0]
                    require(
                        attr.name == "value"
                        and attr.type == onnx.AttributeProto.TENSOR,
                        f"Expected tensor shape Constant: {name}",
                    )
                    tensor = attr.t
                    require(
                        tensor.data_type == onnx.TensorProto.INT64
                        and tensor.data_location != onnx.TensorProto.EXTERNAL
                        and math.prod(tensor.dims) <= 64,
                        f"Expected small inline INT64 shape constant: {name}",
                    )
                    result = onnx.numpy_helper.to_array(tensor)
                else:
                    attrs = attributes(node)
                    if node.op_type == "Shape":
                        require(
                            len(node.input) == 1 and set(attrs) <= {"start", "end"},
                            f"Invalid Shape: {name}",
                        )
                        origin, shape, _, _ = self.resolve(node.input[0])
                        require(origin == source, f"Cross-source reshape shape: {name}")
                        result = np.asarray(shape, dtype=np.int64)[
                            attrs.get("start", 0) : attrs.get("end", len(shape))
                        ]
                    elif node.op_type == "Gather":
                        require(
                            len(node.input) == 2 and set(attrs) <= {"axis"},
                            f"Invalid shape Gather: {name}",
                        )
                        data, indices = [
                            self.constant(value, source, trace) for value in node.input
                        ]
                        require(
                            data.ndim == 1 and attrs.get("axis", 0) in (0, -1),
                            f"Invalid shape Gather axis/rank: {name}",
                        )
                        require(
                            np.all(indices >= -data.size)
                            and np.all(indices < data.size),
                            f"Invalid shape Gather indices: {name}",
                        )
                        result = np.take(data, indices, axis=attrs.get("axis", 0))
                    else:
                        require(
                            bool(node.input)
                            and set(attrs) == {"axis"}
                            and attrs["axis"] in (0, -1),
                            f"Invalid shape Concat: {name}",
                        )
                        values = [
                            self.constant(value, source, trace) for value in node.input
                        ]
                        require(
                            all(value.ndim == 1 for value in values),
                            f"Invalid shape Concat rank: {name}",
                        )
                        require(
                            sum(value.size for value in values) <= 64,
                            f"Oversized shape expression: {name}",
                        )
                        result = np.concatenate(values)
                trace.append(
                    {"name": node.name, "op_type": node.op_type, "output": name}
                )
            require(
                result.ndim <= 1 and result.size <= 64,
                f"Invalid shape expression size/rank: {name}",
            )
            self.check_metadata(name, result.shape, onnx.TensorProto.INT64)
            return result
        finally:
            self.active.remove(key)

    def resolve(self, name):
        key = ("data", name)
        require(key not in self.active, f"Cyclic static transformation: {name}")
        self.active.add(key)
        try:
            require(name not in self.inputs, f"Runtime-dependent expert input: {name}")
            if name in self.tensors:
                tensor = self.tensors[name]
                shape, dtype = tuple(tensor.dims), tensor.data_type
                require(
                    all(dim > 0 for dim in shape), f"Invalid expert dimensions: {name}"
                )
                source, transforms = name, []
            else:
                node = self.node(name, {"Reshape", "Transpose"})
                attrs = attributes(node)
                require(
                    len(node.input) == (2 if node.op_type == "Reshape" else 1),
                    f"Invalid static inputs: {name}",
                )
                source, before, dtype, transforms = self.resolve(node.input[0])
                entry = {
                    "name": node.name,
                    "op_type": node.op_type,
                    "input": node.input[0],
                    "output": name,
                    "input_shape": list(before),
                }
                if node.op_type == "Transpose":
                    require(
                        set(attrs) <= {"perm"}, f"Invalid Transpose attributes: {name}"
                    )
                    perm = attrs.get("perm", list(reversed(range(len(before)))))
                    require(
                        sorted(perm) == list(range(len(before))),
                        f"Invalid Transpose permutation: {name}",
                    )
                    shape = tuple(before[index] for index in perm)
                    entry["perm"] = perm
                else:
                    require(
                        set(attrs) <= {"allowzero"}
                        and attrs.get("allowzero", 0) in (0, 1),
                        f"Invalid Reshape attributes: {name}",
                    )
                    trace = []
                    target = self.constant(node.input[1], source, trace)
                    require(target.ndim == 1, f"Reshape shape must be a vector: {name}")
                    shape = [int(dim) for dim in target]
                    require(
                        all(dim >= -1 for dim in shape) and shape.count(-1) <= 1,
                        f"Invalid Reshape dimensions: {name}",
                    )
                    for index, dim in enumerate(shape):
                        if dim == 0 and not attrs.get("allowzero", 0):
                            require(
                                index < len(before),
                                f"Invalid Reshape zero axis: {name}",
                            )
                            shape[index] = before[index]
                    require(0 not in shape, f"Empty expert Reshape: {name}")
                    known = math.prod(dim for dim in shape if dim != -1)
                    size = math.prod(before)
                    if -1 in shape:
                        require(
                            known > 0 and size % known == 0,
                            f"Invalid inferred Reshape dimension: {name}",
                        )
                        shape[shape.index(-1)] = size // known
                    require(
                        math.prod(shape) == size,
                        f"Reshape changes element count: {name}",
                    )
                    shape = tuple(shape)
                    entry.update(
                        {
                            "shape_input": node.input[1],
                            "shape_values": target.tolist(),
                            "shape_expression": trace,
                            "allowzero": attrs.get("allowzero", 0),
                        }
                    )
                entry.update(
                    {
                        "shape": list(shape),
                        "dtype": onnx.TensorProto.DataType.Name(dtype),
                    }
                )
                transforms = [*transforms, entry]
            self.check_metadata(name, shape, dtype)
            return source, shape, dtype, transforms
        finally:
            self.active.remove(key)


def audit_graph(model):
    nodes = list(model.graph.node)
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    require(len(tensors) == len(model.graph.initializer), "Duplicate initializer names")
    require(
        not any(".experts." in name for name in tensors), "Unfused expert initializer"
    )
    resolver = StaticExpertResolver(model, tensors)
    experts = [node for node in nodes if node.op_type == "QMoE"]
    require(len(experts) == 48, f"Expected 48 QMoE nodes, got {len(experts)}")
    layers = {}
    for node in experts:
        require(
            node.domain == "com.microsoft", f"Unexpected QMoE domain: {node.domain}"
        )
        attrs = attributes(node)
        for name, expected in {
            "weights_prepacked": 0,
            "block_size": 64,
            "k": 8,
            "swiglu_fusion": 1,
        }.items():
            require(
                attrs.get(name) == expected, f"{node.name}: expected {name}={expected}"
            )
        require(
            attrs.get("quant_type", "int") == "int",
            f"{node.name}: expected integer quantization",
        )
        for projection in ("fc1", "fc2"):
            require(
                attrs.get(
                    f"{projection}_expert_weight_bits", attrs.get("expert_weight_bits")
                )
                == 3,
                f"{node.name}: expected {projection} bit3",
            )
        bound_layers = set()
        bindings = {}
        for slot, (suffix, shape, dtype) in EXPERT_INPUTS.items():
            name = node.input[slot] if slot < len(node.input) else ""
            source, resolved_shape, resolved_dtype, transforms = resolver.resolve(name)
            match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\." + suffix, source)
            require(
                match is not None,
                f"{node.name}: cannot infer layer from input {slot}: {name}",
            )
            bound_layers.add(int(match.group(1)))
            require(
                resolved_shape == shape and resolved_dtype == dtype,
                f"Incorrect shape/dtype: {name}: {list(resolved_shape)}, {resolved_dtype}",
            )
            bindings[suffix] = {
                "name": name,
                "source_initializer": source,
                "source_shape": list(tensors[source].dims),
                "transformations": transforms,
                "shape": list(shape),
                "dtype": onnx.TensorProto.DataType.Name(dtype),
            }
        require(len(bound_layers) == 1, f"Cross-layer QMoE inputs: {node.name}")
        layer = bound_layers.pop()
        require(layer not in layers, f"Duplicate QMoE layer: {layer}")
        present = {
            label: node.input[slot]
            for slot, label in ABSENT_INPUTS.items()
            if slot < len(node.input) and node.input[slot]
        }
        require(
            not present,
            f"{node.name}: unexpected explicit bias/zero-point/fc3 inputs: {present}",
        )
        layers[layer] = {
            "layer": layer,
            "name": node.name,
            "attributes": attrs,
            "inputs": list(node.input),
            "tensors": bindings,
            "explicit_bias_zero_points_fc3": present,
        }
    require(set(layers) == set(range(48)), "QMoE layers must cover 0..47 exactly")
    attention, heads, embeddings = {}, [], []
    widths = Counter()
    for node in nodes:
        if node.op_type not in ("MatMulNBits", "GatherBlockQuantized"):
            continue
        require(
            node.domain == "com.microsoft",
            f"Unexpected quantized operator domain: {node.domain}",
        )
        attrs = attributes(node)
        identity = ".".join([node.name, *node.input]).replace("/", ".")
        entry = {"name": node.name, "inputs": list(node.input), "attributes": attrs}
        if node.op_type == "GatherBlockQuantized":
            for key in ("bits", "block_size"):
                if key in attrs:
                    require(
                        attrs[key] == {"bits": 4, "block_size": 64}[key],
                        f"Embedding {key}: {attrs[key]}",
                    )
            embeddings.append(entry)
            continue
        widths[attrs.get("bits")] += 1
        matches = set(
            re.findall(
                r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)(?=\.|$)",
                identity,
            )
        )
        if matches:
            require(len(matches) == 1, f"Ambiguous attention identity: {node.name}")
            key = matches.pop()
            require(key not in attention, f"Duplicate attention projection: {key}")
            require(
                attrs.get("bits") == 4 and attrs.get("block_size") == 64,
                f"Expected INT4/block64 attention: {node.name}",
            )
            attention[key] = entry
        elif re.search(r"(?:^|\.)lm_head(?:\.|$)", identity):
            require(
                attrs.get("bits") == 8 and attrs.get("block_size") == 64,
                "Expected INT8/block64 head",
            )
            heads.append(entry)
        else:
            raise ValueError(f"Unidentified MatMulNBits node: {node.name}")
    expected_attention = {
        (str(layer), projection)
        for layer in range(48)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    require(
        set(attention) == expected_attention and len(heads) == 1,
        "Expected 192 attention projections and one head",
    )
    return {
        "qmoe_nodes": len(experts),
        "layers": [layers[layer] for layer in sorted(layers)],
        "linear_width_counts": dict(widths),
        "attention": list(attention.values()),
        "head": heads,
        "embeddings": embeddings,
        "node_counts": dict(Counter(node.op_type for node in nodes)),
    }


def audit(root: Path):
    root = root.resolve()
    graph = safe_file(root, "model.onnx")
    model = onnx.load(graph, load_external_data=False)
    report = audit_graph(model)
    ranges = external_ranges(model, root)
    locations = {"model.onnx", *(entry["location"] for entry in ranges)}
    for pattern in (
        "tokenizer*",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab*",
        "merges*",
        "chat_template*",
        "*.model",
        "*config.json",
    ):
        locations.update(
            path.relative_to(root).as_posix()
            for path in root.rglob(pattern)
            if path.is_file()
        )
    files = {}
    for location in sorted(locations):
        path = safe_file(root, location)
        files[location] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    report.update(
        {
            "valid": True,
            "root": str(root),
            "external_ranges": ranges,
            "files": files,
            "scope": "Metadata, static initializer transformation traces and streaming file hashes only; "
            "swiglu_fusion=1 declares interleaving. "
            "Packed weight contents, scale values and isolated-layer numerical correctness are not verified.",
        }
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Metadata-only Qwen3-30B-A3B INT3 artifact audit"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", "--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.root)
    except (
        ValueError,
        OSError,
        onnx.onnx_cpp2py_export.checker.ValidationError,
    ) as error:
        report = {"valid": False, "error": str(error)}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
