"""Production packaging helpers for optimized CLM Foundry Local artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper
from tokenizers import Tokenizer

MAGIC = b"CLMACT1\0"


def _replace_names(model: onnx.ModelProto, replacements: dict[str, str]) -> None:
    for node in model.graph.node:
        node.input[:] = [replacements.get(name, name) for name in node.input]
        node.output[:] = [replacements.get(name, name) for name in node.output]
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        value.name = replacements.get(value.name, value.name)


def _external_location(initializer: onnx.TensorProto, location: str) -> None:
    for entry in initializer.external_data:
        if entry.key == "location":
            entry.value = location


def build_fused_clm(package: Path) -> None:
    output = package / "fused_state_ranking.onnx"
    encoder = onnx.load(package / "encoder" / "model.onnx", load_external_data=False)
    state = compose.add_prefix(
        onnx.load(package / "state_head" / "model.onnx", load_external_data=False),
        "state/",
    )
    scorer = compose.add_prefix(
        onnx.load(package / "scorer" / "model.onnx", load_external_data=False),
        "scorer/",
    )
    encoder_type = encoder.graph.output[0].type.tensor_type.elem_type
    for initializer in encoder.graph.initializer:
        _external_location(
            initializer,
            os.path.relpath(package / "encoder" / "model.onnx.data", output.parent),
        )
    for initializer in state.graph.initializer:
        _external_location(
            initializer,
            os.path.relpath(package / "state_head" / "model.onnx.data", output.parent),
        )
    _replace_names(state, {"state/embeddings": "fused/pooled_embeddings"})
    _replace_names(
        scorer,
        {
            "scorer/state_projections": "state/projections",
            "scorer/action_projections": "action_projections",
            "scorer/temperature": "temperature",
            "scorer/candidate_owners": "candidate_owners",
            "scorer/logits": "logits",
            "scorer/probabilities": "probabilities",
        },
    )
    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "fused/axis1"),
        numpy_helper.from_array(np.array(1e-12, dtype=np.float32), "fused/epsilon"),
    ]
    nodes = [
        helper.make_node(
            "Mul",
            ["token_hidden_states", "last_token_selector"],
            ["fused/selected_hidden"],
        ),
        helper.make_node(
            "ReduceSum",
            ["fused/selected_hidden", "fused/axis1"],
            ["fused/last_hidden_low_precision"],
            keepdims=0,
        ),
        helper.make_node(
            "Cast",
            ["fused/last_hidden_low_precision"],
            ["fused/last_hidden"],
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "ReduceL2",
            ["fused/last_hidden", "fused/axis1"],
            ["fused/norm"],
            keepdims=1,
        ),
        helper.make_node(
            "Clip",
            ["fused/norm", "fused/epsilon"],
            ["fused/safe_norm"],
        ),
        helper.make_node(
            "Div",
            ["fused/last_hidden", "fused/safe_norm"],
            ["fused/pooled_embeddings"],
        ),
    ]
    scorer_inputs = [
        copy.deepcopy(value)
        for value in scorer.graph.input
        if value.name in {"action_projections", "temperature", "candidate_owners"}
    ]
    outputs = {value.name: value for value in scorer.graph.output}
    graph = helper.make_graph(
        [*encoder.graph.node, *nodes, *state.graph.node, *scorer.graph.node],
        "clm_fused_state_ranking",
        [
            *copy.deepcopy(encoder.graph.input),
            helper.make_tensor_value_info(
                "last_token_selector",
                encoder_type,
                ["batch", "sequence_length", 1],
            ),
            *scorer_inputs,
        ],
        [
            copy.deepcopy(outputs["probabilities"]),
            copy.deepcopy(outputs["logits"]),
            copy.deepcopy(state.graph.output[0]),
        ],
        [
            *copy.deepcopy(encoder.graph.initializer),
            *initializers,
            *copy.deepcopy(state.graph.initializer),
            *copy.deepcopy(scorer.graph.initializer),
        ],
        value_info=[
            *copy.deepcopy(encoder.graph.value_info),
            *copy.deepcopy(state.graph.value_info),
            *copy.deepcopy(scorer.graph.value_info),
        ],
    )
    fused = helper.make_model(
        graph,
        opset_imports=copy.deepcopy(encoder.opset_import),
        functions=copy.deepcopy(encoder.functions),
        producer_name="olive-system-one",
    )
    fused.ir_version = encoder.ir_version
    onnx.save(fused, output)
    onnx.checker.check_model(output)


def precompute_actions(
    package: Path, catalog_path: Path, provider: str
) -> None:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("fixed_action_catalog must be a non-empty id-to-text object")
    texts = list(catalog.values())
    if not all(isinstance(text, str) and text for text in texts):
        raise ValueError("fixed action texts must be non-empty strings")
    tokenizer = Tokenizer.from_file(str(package / "tokenizer.json"))
    config = json.loads((package / "tokenizer_config.json").read_text())
    pad = tokenizer.token_to_id(config["pad_token"])
    rows = [tokenizer.encode(text).ids[:2048] for text in texts]
    width = max(map(len, rows))
    ids = np.full((len(rows), width), pad, dtype=np.int64)
    mask = np.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = row
        mask[index, : len(row)] = 1
    positions = np.maximum(np.cumsum(mask, axis=1) - 1, 0).astype(np.int64)
    session = ort.InferenceSession(
        str(package / "encoder" / "model.onnx"),
        providers=[provider],
    )
    input_names = {value.name for value in session.get_inputs()}
    required_inputs = {"input_ids", "attention_mask"}
    if not required_inputs.issubset(input_names):
        raise ValueError(
            "fixed action encoder must declare input_ids and attention_mask"
        )
    unsupported_inputs = input_names - {
        "input_ids",
        "attention_mask",
        "position_ids",
    }
    if unsupported_inputs:
        raise ValueError(
            f"fixed action encoder has unsupported inputs: {sorted(unsupported_inputs)}"
        )
    encoder_feeds = {"input_ids": ids, "attention_mask": mask}
    if "position_ids" in input_names:
        encoder_feeds["position_ids"] = positions
    hidden = session.run(
        ["token_hidden_states"],
        encoder_feeds,
    )[0]
    if not np.isfinite(hidden).all():
        raise ValueError("fixed action encoder output is non-finite")
    pooled = hidden[np.arange(len(rows)), mask.sum(axis=1) - 1].astype(np.float32)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    action = ort.InferenceSession(
        str(package / "action_head" / "model.onnx"),
        providers=[provider],
    )
    values = action.run(["projections"], {"embeddings": pooled})[0].astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("fixed action projections are non-finite")
    with (package / "precomputed_action_projections.bin").open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(catalog)))
        for text, projection in zip(texts, values, strict=True):
            encoded = text.encode()
            stream.write(struct.pack("<II", len(encoded), len(projection)))
            stream.write(encoded)
            stream.write(projection.astype("<f4", copy=False).tobytes())
    metadata = {
        "schema_version": 1,
        "format": "CLMACT1",
        "count": len(catalog),
        "projection_size": int(values.shape[1]),
        "catalog": [
            {"id": key, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            for (key, text) in catalog.items()
        ],
    }
    (package / "precomputed_action_projections.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def derive_buckets(package: Path, readiness_path: Path) -> None:
    texts = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not isinstance(texts, list) or not texts or not all(
        isinstance(text, str) and text for text in texts
    ):
        raise ValueError("readiness_texts must be a non-empty string array")
    tokenizer = Tokenizer.from_file(str(package / "tokenizer.json"))
    widths = [len(tokenizer.encode(text).ids) for text in texts]
    rounded = {((width + 7) // 8) * 8 for width in widths}
    buckets = sorted(rounded | {value for value in (256, 512, 1024, 2048) if value > max(widths)})
    (package / "clm_cuda_graph_buckets.txt").write_text(
        "\n".join(map(str, buckets)) + "\n"
    )
    (package / "clm_cuda_graph_buckets.json").write_text(
        json.dumps(
            {"request_token_widths": widths, "rounding_multiple": 8, "buckets": buckets},
            indent=2,
        )
        + "\n"
    )


def finalize_clm_package(
    package: Path, options: dict, execution_provider: str
) -> None:
    expected = {"fallback_idle_ms", "fixed_action_catalog", "readiness_texts"}
    if set(options) != expected:
        raise ValueError(
            f"runtime_optimizations must contain exactly {sorted(expected)}"
        )
    if execution_provider != "cuda":
        raise ValueError("optimized CLM packaging currently requires CUDA")
    if not (package / "safe_encoder" / "model.onnx").is_file():
        raise ValueError("optimized CLM package is missing its safe encoder")
    build_fused_clm(package)
    precompute_actions(
        package,
        Path(options["fixed_action_catalog"]),
        "CUDAExecutionProvider",
    )
    derive_buckets(package, Path(options["readiness_texts"]))
    fallback_idle_ms = options["fallback_idle_ms"]
    if not isinstance(fallback_idle_ms, int) or isinstance(
        fallback_idle_ms, bool
    ) or fallback_idle_ms < 0:
        raise ValueError("fallback_idle_ms must be a non-negative integer")
    (package / "clm_fallback_idle_ms.txt").write_text(
        f"{fallback_idle_ms}\n", encoding="ascii"
    )
    manifest_path = package / "component_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["components"].update(
        {
            "fused_state_ranking": {
                "role": "fused",
                "filename": "fused_state_ranking.onnx",
            },
            "safe_encoder": {
                "role": "numerical_fallback",
                "filename": "safe_encoder/model.onnx",
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    metadata_path = package / "inference_model.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["Provider"] = {
        "execution_provider": "cuda",
        "variant": "fp16-bf16-safety-fallback",
    }
    metadata["Capabilities"] = sorted(
        set(metadata["Capabilities"])
        | {
            "precomputed-actions",
            "fused-ranking",
            "cuda-graph",
            "finite-output-validation",
            "numerical-fallback",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
