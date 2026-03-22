# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Custom Olive pass: NemotronExport.

Wraps scripts/export_nemotron_to_onnx_static_shape.py so that the NeMo
model export step can be executed as the first pass in an Olive workflow
run, feeding directly into the graph-fusion and INT4-quantization passes.
"""

import subprocess
import sys
from pathlib import Path

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam


class NemotronExport(Pass):
    """Export the Nemotron Speech Streaming NeMo model to ONNX.

    Runs scripts/export_nemotron_to_onnx_static_shape.py via subprocess and
    returns an ONNXModelHandler pointing to the exported encoder so that
    subsequent Olive passes (OrtTransformersOptimization, OnnxBlockWiseRtnQuantization)
    can operate on it.

    The decoder, joint network, and config files produced alongside the encoder
    are written to the same Olive-managed output directory. They are not consumed
    by later passes but remain available for the user to copy to the final
    artifact directory after the workflow completes.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict:
        return {
            "streaming": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Export in streaming mode with cache-aware static-shape I/O.",
            ),
            "chunk_size": PassConfigParam(
                type_=float,
                default_value=0.56,
                description="Streaming chunk size in seconds (0.08, 0.16, 0.56, or 1.12).",
            ),
            "left_chunks": PassConfigParam(
                type_=int,
                default_value=10,
                description="Number of left context chunks for streaming attention.",
            ),
            "device": PassConfigParam(
                type_=str,
                default_value="cpu",
                description="Device to use during NeMo export ('cpu' or 'cuda').",
            ),
        }

    def _run_for_config(self, model, config, output_model_path: str) -> ONNXModelHandler:
        # Resolve the output path to a concrete encoder.onnx file path
        # so it matches the filename the export script produces.
        output_model_path = resolve_onnx_path(output_model_path, "encoder.onnx")
        output_dir = Path(output_model_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # The export script lives at scripts/ one level above cpu/
        script = Path(__file__).parent.parent / "scripts" / "export_nemotron_to_onnx_static_shape.py"

        cmd = [
            sys.executable, str(script),
            "--model_name", model.model_path,
            "--output_dir", str(output_dir),
            "--chunk_size", str(config.chunk_size),
            "--left_chunks", str(config.left_chunks),
            "--device", config.device,
        ]
        if config.streaming:
            cmd.append("--streaming")

        subprocess.run(cmd, check=True)

        return ONNXModelHandler(model_path=output_model_path)
