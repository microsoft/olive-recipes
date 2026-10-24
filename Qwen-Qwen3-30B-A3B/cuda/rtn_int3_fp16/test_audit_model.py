import hashlib
import json

import onnx
import pytest
from onnx import TensorProto, helper

import audit_model


def model_fixture():
    nodes, tensors = [], []
    for layer in range(48):
        inputs = ["hidden", "router", "", "", "", "", ""]
        for slot, (suffix, shape, dtype) in audit_model.EXPERT_INPUTS.items():
            name = f"model.layers.{layer}.mlp.{suffix}"
            inputs[slot] = name
            tensors.append(TensorProto(name=name, dims=shape, data_type=dtype))
        nodes.append(
            helper.make_node(
                "QMoE",
                inputs,
                [f"out{layer}"],
                name=f"moe{layer}",
                domain="com.microsoft",
                expert_weight_bits=3,
                block_size=64,
                weights_prepacked=0,
                k=8,
                swiglu_fusion=1,
            )
        )
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    ["hidden", f"model.layers.{layer}.self_attn.{projection}.weight"],
                    [f"attention{layer}{projection}"],
                    domain="com.microsoft",
                    bits=4,
                    block_size=64,
                )
            )
    nodes.append(
        helper.make_node(
            "MatMulNBits",
            ["hidden", "lm_head.weight"],
            ["logits"],
            domain="com.microsoft",
            bits=8,
            block_size=64,
        )
    )
    nodes.append(
        helper.make_node(
            "GatherBlockQuantized",
            ["model.embed_tokens.weight", "ids"],
            ["hidden"],
            domain="com.microsoft",
            bits=4,
            block_size=64,
        )
    )
    return helper.make_model(helper.make_graph(nodes, "fixture", [], [], tensors))


def test_shuffled_layers():
    model = model_fixture()
    model.graph.node.reverse()
    report = audit_model.audit_graph(model)
    assert [entry["layer"] for entry in report["layers"]] == list(range(48))
    assert report["linear_width_counts"] == {4: 192, 8: 1}
    assert report["layers"][0]["explicit_bias_zero_points_fc3"] == {}


def interleave_fixture(expression=False):
    model = model_fixture()
    node = model.graph.node[0]
    for slot in (2, 3):
        source = node.input[slot]
        width = 768 if slot == 2 else 32
        first, last = f"first{slot}", f"last{slot}"
        for name, values in ((first, [128, 2, 768, width]), (last, [128, 1536, width])):
            model.graph.initializer.append(
                helper.make_tensor(name, TensorProto.INT64, [len(values)], values)
            )
        if expression:
            model.graph.initializer.pop()
            model.graph.node.append(helper.make_node("Shape", [source], [last]))
            model.graph.initializer.pop()
            model.graph.initializer.extend(
                [
                    helper.make_tensor(f"indices{slot}", TensorProto.INT64, [1], [0]),
                    helper.make_tensor(
                        f"tail{slot}", TensorProto.INT64, [3], [2, 768, width]
                    ),
                ]
            )
            model.graph.node.extend(
                [
                    helper.make_node(
                        "Gather", [last, f"indices{slot}"], [f"batch{slot}"], axis=0
                    ),
                    helper.make_node(
                        "Concat", [f"batch{slot}", f"tail{slot}"], [first], axis=0
                    ),
                ]
            )
        model.graph.node.extend(
            [
                helper.make_node("Reshape", [source, first], [f"split{slot}"]),
                helper.make_node(
                    "Transpose",
                    [f"split{slot}"],
                    [f"transpose{slot}"],
                    perm=[0, 2, 1, 3],
                ),
                helper.make_node(
                    "Reshape", [f"transpose{slot}", last], [f"interleaved{slot}"]
                ),
            ]
        )
        node.input[slot] = f"interleaved{slot}"
    return model


@pytest.mark.parametrize("expression", [False, True])
def test_static_interleave(expression):
    model = interleave_fixture(expression)
    model.graph.node.reverse()
    report = audit_model.audit_graph(model)
    for suffix, width in (("fc1_experts_weights", 768), ("fc1_scales", 32)):
        binding = report["layers"][0]["tensors"][suffix]
        assert binding["source_initializer"] == f"model.layers.0.mlp.{suffix}"
        assert binding["shape"] == [128, 1536, width]
        assert [entry["op_type"] for entry in binding["transformations"]] == [
            "Reshape",
            "Transpose",
            "Reshape",
        ]
        assert binding["transformations"][1]["shape"] == [128, 768, 2, width]
        assert bool(binding["transformations"][0]["shape_expression"]) == expression


@pytest.mark.parametrize(
    "defect",
    [
        "crosslayer",
        "runtime_shape",
        "runtime_source",
        "operator",
        "perm",
        "reshape",
        "metadata",
        "shape_crosslayer",
    ],
)
def test_static_interleave_rejects(defect):
    model = interleave_fixture(expression=True)
    split = next(node for node in model.graph.node if list(node.output) == ["split2"])
    transpose = next(
        node for node in model.graph.node if list(node.output) == ["transpose2"]
    )
    if defect == "crosslayer":
        split.input[0] = "model.layers.1.mlp.fc1_experts_weights"
        next(node for node in model.graph.node if list(node.output) == ["last2"]).input[
            0
        ] = split.input[0]
    elif defect == "runtime_shape":
        split.input[1] = "router"
    elif defect == "runtime_source":
        split.input[0] = "hidden"
    elif defect == "operator":
        transpose.op_type = "Cast"
    elif defect == "perm":
        transpose.attribute[0].ints[:] = [0, 2, 2, 3]
    elif defect == "reshape":
        tensor = next(
            tensor for tensor in model.graph.initializer if tensor.name == "tail2"
        )
        tensor.int64_data[1] = 767
    elif defect == "metadata":
        model.graph.value_info.append(
            helper.make_tensor_value_info(
                "interleaved2", TensorProto.UINT8, [128, 1536, 767]
            )
        )
    else:
        next(node for node in model.graph.node if list(node.output) == ["last2"]).input[
            0
        ] = "model.layers.1.mlp.fc1_experts_weights"
    with pytest.raises(ValueError):
        audit_model.audit_graph(model)


def test_static_interleave_constant_and_inferred_dimensions(monkeypatch):
    model = interleave_fixture()
    first = next(
        tensor for tensor in model.graph.initializer if tensor.name == "first2"
    )
    first.int64_data[:] = [0, 2, -1, 768]
    constant = TensorProto()
    constant.CopyFrom(first)
    model.graph.initializer.remove(first)
    model.graph.node.append(
        helper.make_node("Constant", [], ["first2"], value=constant)
    )
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                "split2", TensorProto.UINT8, [128, 2, 768, 768]
            ),
            helper.make_tensor_value_info(
                "interleaved2", TensorProto.UINT8, [128, "channels", 768]
            ),
        ]
    )
    original = onnx.numpy_helper.to_array

    def small_inline_only(tensor, *args, **kwargs):
        assert tensor.data_type == TensorProto.INT64
        assert len(tensor.dims) <= 1
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(onnx.numpy_helper, "to_array", small_inline_only)
    model.graph.node.append(
        helper.make_node("UnsupportedRuntimeRouter", ["runtime"], ["router"])
    )
    assert audit_model.audit_graph(model)["qmoe_nodes"] == 48


@pytest.mark.parametrize(
    "defect",
    [
        "external_shape",
        "float_shape",
        "duplicate_negative",
        "bad_negative",
        "zero",
        "gather",
        "concat",
        "cycle",
        "metadata_dtype",
        "runtime_override",
    ],
)
def test_static_shape_rejects(defect):
    model = interleave_fixture(expression=defect in ("gather", "concat"))
    if defect in (
        "external_shape",
        "float_shape",
        "duplicate_negative",
        "bad_negative",
        "zero",
    ):
        tensor = next(
            tensor for tensor in model.graph.initializer if tensor.name == "first2"
        )
        if defect == "external_shape":
            tensor.data_location = TensorProto.EXTERNAL
        elif defect == "float_shape":
            tensor.data_type = TensorProto.FLOAT
        else:
            tensor.int64_data[:] = {
                "duplicate_negative": [128, -1, -1, 768],
                "bad_negative": [128, 2, -2, 768],
                "zero": [128, 2, 768, 0],
            }[defect]
    elif defect == "gather":
        tensor = next(
            tensor for tensor in model.graph.initializer if tensor.name == "indices2"
        )
        tensor.int64_data[0] = 3
    elif defect == "concat":
        node = next(
            node for node in model.graph.node if list(node.output) == ["first2"]
        )
        node.attribute[0].i = 1
    elif defect == "cycle":
        node = next(
            node for node in model.graph.node if list(node.output) == ["split2"]
        )
        node.input[0] = "interleaved2"
    elif defect == "metadata_dtype":
        model.graph.value_info.append(
            helper.make_tensor_value_info(
                "interleaved2", TensorProto.FLOAT16, [128, 1536, 768]
            )
        )
    else:
        model.graph.input.append(
            helper.make_tensor_value_info(
                "model.layers.0.mlp.fc1_experts_weights",
                TensorProto.UINT8,
                [128, 1536, 768],
            )
        )
    with pytest.raises(ValueError):
        audit_model.audit_graph(model)


@pytest.mark.parametrize(
    "defect",
    [
        "crosswire",
        "duplicate",
        "dtype",
        "shape",
        "bits",
        "layout",
        "bias",
        "zeros",
        "attention",
        "embedding",
    ],
)
def test_graph_rejects(defect):
    model = model_fixture()
    node = model.graph.node[0]
    if defect == "crosswire":
        node.input[5] = "model.layers.1.mlp.fc2_experts_weights"
    elif defect == "duplicate":
        model.graph.node[5].input[:] = node.input
    elif defect == "dtype":
        model.graph.initializer[1].data_type = TensorProto.FLOAT
    elif defect == "shape":
        model.graph.initializer[0].dims[2] += 1
    elif defect in ("bits", "layout"):
        key = {"bits": "expert_weight_bits", "layout": "swiglu_fusion"}[defect]
        next(attr for attr in node.attribute if attr.name == key).i = 2
    elif defect in ("bias", "zeros"):
        node.input.extend([""] * 7)
        node.input[4 if defect == "bias" else 11] = "explicit"
    elif defect == "attention":
        next(
            attr for attr in model.graph.node[1].attribute if attr.name == "bits"
        ).i = 8
    else:
        next(
            attr for attr in model.graph.node[-1].attribute if attr.name == "bits"
        ).i = 8
    with pytest.raises(ValueError):
        audit_model.audit_graph(model)


def external_tensor(location="weights.bin", offset="0", length="4"):
    tensor = TensorProto(
        name="small",
        dims=[2],
        data_type=TensorProto.FLOAT16,
        data_location=TensorProto.EXTERNAL,
    )
    for key, value in {
        "location": location,
        "offset": offset,
        "length": length,
    }.items():
        tensor.external_data.add(key=key, value=value)
    return tensor


def test_external_ranges_and_stream_hash(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abcd")
    model = helper.make_model(
        helper.make_graph([], "external", [], [], [external_tensor()])
    )
    assert audit_model.external_ranges(model, tmp_path)[0]["length"] == 4
    assert audit_model.sha256_file(path) == hashlib.sha256(b"abcd").hexdigest()
    model.graph.initializer[0].external_data.pop()
    assert audit_model.external_ranges(model, tmp_path)[0]["length"] == 4


@pytest.mark.parametrize(
    "location,offset,length",
    [
        ("../escape", "0", "4"),
        ("/etc/passwd", "0", "4"),
        ("C:\\weights.bin", "0", "4"),
        ("missing", "0", "4"),
        ("weights.bin", "-1", "4"),
        ("weights.bin", "1", "4"),
        ("weights.bin", "0", "3"),
        ("weights.bin", "invalid", "4"),
    ],
)
def test_bad_external_ranges(tmp_path, location, offset, length):
    (tmp_path / "weights.bin").write_bytes(b"abcd")
    model = helper.make_model(
        helper.make_graph(
            [], "external", [], [], [external_tensor(location, offset, length)]
        )
    )
    with pytest.raises(ValueError):
        audit_model.external_ranges(model, tmp_path)


def test_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside").write_bytes(b"abcd")
    (root / "weights.bin").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        audit_model.safe_file(root, "weights.bin")


def test_audit_metadata_only_and_cli(tmp_path, monkeypatch, capsys):
    model = model_fixture()
    onnx.save(model, tmp_path / "model.onnx")
    (tmp_path / "tokenizer.json").write_text("{}")
    original = onnx.load

    def metadata_load(path, *, load_external_data):
        assert load_external_data is False
        return original(path, load_external_data=False)

    monkeypatch.setattr(onnx, "load", metadata_load)
    report = audit_model.audit(tmp_path)
    assert set(report["files"]) == {"model.onnx", "tokenizer.json"}
    output = tmp_path / "report.json"
    assert audit_model.main([str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["valid"]
    assert json.loads(capsys.readouterr().out)["valid"]


def test_cli_failure(tmp_path, capsys):
    assert audit_model.main([str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_effective_projection_bits_and_optional_embedding_attributes():
    model = model_fixture()
    for node in model.graph.node:
        if node.op_type == "QMoE":
            next(
                attr for attr in node.attribute if attr.name == "expert_weight_bits"
            ).i = 4
            node.attribute.extend(
                [
                    helper.make_attribute("fc1_expert_weight_bits", 3),
                    helper.make_attribute("fc2_expert_weight_bits", 3),
                ]
            )
    model.graph.node[-1].ClearField("attribute")
    assert audit_model.audit_graph(model)["qmoe_nodes"] == 48
    model.graph.node[0].attribute[-1].i = 4
    with pytest.raises(ValueError, match="fc2 bit3"):
        audit_model.audit_graph(model)


@pytest.mark.parametrize(
    "attribute", ["weights_prepacked", "block_size", "k", "swiglu_fusion"]
)
def test_required_attributes(tmp_path, attribute):
    model = model_fixture()
    node = model.graph.node[0]
    next(attr for attr in node.attribute if attr.name == attribute).i = -1
    with pytest.raises(ValueError, match=attribute):
        audit_model.audit_graph(model)


def test_nested_external_and_all_file_hashes(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"abcd")
    model = model_fixture()
    model.graph.node.append(
        helper.make_node("Constant", [], ["constant"], value=external_tensor())
    )
    onnx.save(model, tmp_path / "model.onnx")
    (tmp_path / "tokenizer.json").write_text("{}")
    report = audit_model.audit(tmp_path)
    assert report["external_ranges"] == [
        {"tensor": "small", "location": "weights.bin", "offset": 0, "length": 4}
    ]
    assert set(report["files"]) == {"model.onnx", "weights.bin", "tokenizer.json"}
    assert (
        report["files"]["weights.bin"]["sha256"] == hashlib.sha256(b"abcd").hexdigest()
    )


def test_streaming_hash_across_chunks(tmp_path):
    content = b"a" * (2 * 1024 * 1024 + 3)
    path = tmp_path / "chunks.bin"
    path.write_bytes(content)
    assert audit_model.sha256_file(path) == hashlib.sha256(content).hexdigest()
