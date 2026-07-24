"""Standalone exporter: HuggingFace MoonshineStreaming -> five stateful ONNX
graphs for onnxruntime-genai (frontend / encoder / adapter / cross_kv /
decoder_kv).

Run inside a Python env with the moonshine transformers integration installed:

    conda activate moonshine   # or: source .venv/bin/activate
    python export_moonshine_streaming.py \
        --model usefulsensors/moonshine-streaming-tiny \
        --output-dir build/moonshine-streaming-tiny

The exporter uses the TorchDynamo ONNX path (``dynamo=True``) so data-dependent
shapes in the stateful frontend export cleanly, then rewrites every graph's
input/output names to the exact contract genai expects.
"""

from __future__ import annotations

import argparse
import os

import onnx
import torch
from torch.export import Dim

import moonshine_model_load as mml

AUTO = Dim.AUTO


# --------------------------------------------------------------------------- #
# Per-component export specification                                          #
# --------------------------------------------------------------------------- #
def build_specs():
    """Return the ordered list of component export specs.  ``dynamic`` is a
    tuple aligned with the positional dummy inputs: each entry is either a
    ``{axis: Dim.AUTO}`` dict or ``None`` for a fully static input."""
    return [
        {
            "name": "frontend",
            "loader": mml.frontend_model_loader,
            "dummy": mml.frontend_dummy_inputs,
            "input_names": [
                "audio_chunk", "sample_buffer", "sample_len",
                "conv1_buffer", "conv2_buffer", "frame_count",
            ],
            "output_names": [
                "features", "sample_buffer_out", "sample_len_out",
                "conv1_buffer_out", "conv2_buffer_out", "frame_count_out",
            ],
            "dynamic": ({1: AUTO}, None, None, None, None, None),
        },
        {
            "name": "encoder",
            "loader": mml.encoder_model_loader,
            "dummy": mml.encoder_dummy_inputs,
            "input_names": ["features"],
            "output_names": ["encoded"],
            "dynamic": ({1: AUTO},),
        },
        {
            "name": "adapter",
            "loader": mml.adapter_model_loader,
            "dummy": mml.adapter_dummy_inputs,
            "input_names": ["encoded", "pos_offset"],
            "output_names": ["memory"],
            "dynamic": ({1: AUTO}, None),
        },
        {
            "name": "cross_kv",
            "loader": mml.cross_kv_model_loader,
            "dummy": mml.cross_kv_dummy_inputs,
            "input_names": ["memory"],
            "output_names": ["k_cross", "v_cross"],
            "dynamic": ({1: AUTO},),
        },
        {
            "name": "decoder_kv",
            "loader": mml.decoder_kv_model_loader,
            "dummy": mml.decoder_kv_dummy_inputs,
            "input_names": ["token", "k_self", "v_self", "out_k_cross", "out_v_cross"],
            "output_names": [
                "logits", "out_k_self", "out_v_self", "out_k_cross", "out_v_cross",
            ],
            "dynamic": ({1: AUTO}, {3: AUTO}, {3: AUTO}, {3: AUTO}, {3: AUTO}),
        },
    ]


# --------------------------------------------------------------------------- #
# ONNX I/O renaming (make names match the genai contract exactly)             #
# --------------------------------------------------------------------------- #
def rename_io(path, input_names, output_names):
    model = onnx.load(path)
    graph = model.graph

    def remap(value_infos, desired):
        mapping = {}
        for vi, new in zip(value_infos, desired):
            if vi.name != new:
                mapping[vi.name] = new
                vi.name = new
        return mapping

    in_map = remap(graph.input, input_names)
    out_map = remap(graph.output, output_names)
    rename = {**in_map, **out_map}
    if rename:
        for node in graph.node:
            node.input[:] = [rename.get(n, n) for n in node.input]
            node.output[:] = [rename.get(n, n) for n in node.output]
    onnx.save(model, path)
    return [i.name for i in graph.input], [o.name for o in graph.output]


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def export_component(spec, model_name, output_dir, opset):
    name = spec["name"]
    print(f"\n=== exporting {name} ===")
    module = spec["loader"](model_name).eval()
    dummy = spec["dummy"](module)
    # dummy_inputs_func returns a dict keyed by forward-parameter name (Olive's
    # kwargs export path); the standalone exporter uses positional args, so
    # flatten to a value tuple in the declared input order.
    dummy_args = tuple(dummy.values()) if isinstance(dummy, dict) else tuple(dummy)
    onnx_path = os.path.join(output_dir, f"{name}.onnx")

    torch.onnx.export(
        module,
        dummy_args,
        onnx_path,
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        dynamic_shapes=spec["dynamic"],
        opset_version=opset,
        dynamo=True,
    )
    ins, outs = rename_io(onnx_path, spec["input_names"], spec["output_names"])
    onnx.checker.check_model(onnx_path)
    print(f"  inputs : {ins}")
    print(f"  outputs: {outs}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="usefulsensors/moonshine-streaming-tiny")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--chunk-samples", type=int, default=8000)
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Optional subset of component names to export.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mml.set_chunk_samples(args.chunk_samples)

    specs = build_specs()
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]

    for spec in specs:
        export_component(spec, args.model, args.output_dir, args.opset)

    print(f"\nDone. Graphs written to {args.output_dir}")


if __name__ == "__main__":
    main()
