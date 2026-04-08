# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Custom Olive passes for the Nemotron Speech Streaming recipe.

NemotronExport:
    Exports the NeMo model to ONNX by calling the importable
    export_to_onnx() and export_tokenizer() functions from the scripts/
    directory directly.  The encoder.onnx produced here is then consumed
    by subsequent Olive passes (OrtTransformersOptimization,
    NemotronKQuantQuantization).

NemotronKQuantQuantization:
    Quantizes the fused encoder ONNX model to INT4 using the k-quant
    algorithm from OnnxRuntime's MatMulNBitsQuantizer with
    KQuantWeightOnlyQuantConfig.  This produces an INT4-only encoder with
    all MatMul weights quantized via k-quant instead of RTN.
"""

import sys
from pathlib import Path

import onnx
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam

# scripts/ directory lives one level above the cpu/ directory that contains
# this file.  Adding it to sys.path allows the export helpers to be imported
# without installing them as a package.
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


class NemotronExport(Pass):
    """Export the Nemotron Speech Streaming NeMo model to ONNX.

    Calls scripts/export_nemotron_to_onnx_static_shape.export_to_onnx() and
    scripts/export_tokenizer.export_tokenizer() directly (no subprocess) and
    returns an ONNXModelHandler pointing to the exported encoder so that
    subsequent Olive passes (OrtTransformersOptimization,
    NemotronKQuantQuantization) can operate on it.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict:
        return {
            "model_name": PassConfigParam(
                type_=str,
                default_value="nvidia/nemotron-speech-streaming-en-0.6b",
                description=(
                    "HuggingFace model ID or path to a local .nemo file to export. "
                    "Passed directly to export_to_onnx()."
                ),
            ),
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
        from export_nemotron_to_onnx_static_shape import export_to_onnx  # noqa: PLC0415
        from export_tokenizer import export_tokenizer  # noqa: PLC0415

        # Resolve the output path to a concrete encoder.onnx file path
        # so it matches the filename the export script produces.
        output_model_path = resolve_onnx_path(output_model_path, "encoder.onnx")
        output_dir = Path(output_model_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the device value – Olive's resource-path scanner may coerce
        # bare strings that match local directory names (e.g. "cpu" → ./cpu/).
        # Retrieve the original token from the path name in that case.
        device = config.device
        if Path(str(device)).is_absolute():
            device = Path(str(device)).name

        export_to_onnx(
            model_name=config.model_name,
            output_dir=str(output_dir),
            streaming=config.streaming,
            chunk_size=config.chunk_size,
            left_chunks=config.left_chunks,
            device=device,
        )

        # Export tokenizer files (tokenizer.json, tokenizer_config.json, vocab.txt)
        # to the same output directory so they are available for ORT GenAI.
        export_tokenizer(
            model_name=config.model_name,
            output_dir=str(output_dir),
        )

        return ONNXModelHandler(model_path=output_model_path)


class NemotronKQuantQuantization(Pass):
    """Quantize the Nemotron encoder to INT4 using k-quant (KQuantWeightOnlyQuantConfig).

    Applies the k-quant weight-only quantization algorithm from OnnxRuntime's
    MatMulNBitsQuantizer, producing an INT4-only encoder with all MatMul weights
    quantized via k-quant instead of RTN.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict:
        return {
            "block_size": PassConfigParam(
                type_=int,
                default_value=32,
                description="Block size for weight quantization. Default is 32.",
            ),
            "is_symmetric": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Whether to use symmetric quantization. Default is True.",
            ),
            "accuracy_level": PassConfigParam(
                type_=int,
                default_value=4,
                description=(
                    "Accuracy level for the 4-bit quantized MatMul computation. "
                    "Refer to the MatMulNBits contrib op's 'accuracy_level' attribute. "
                    "Default is 4."
                ),
            ),
        }

    def _run_for_config(self, model, config, output_model_path: str) -> ONNXModelHandler:
        from onnxruntime.quantization.matmul_nbits_quantizer import (  # noqa: PLC0415
            KQuantWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)

        src_model = onnx.load(str(model.model_path), load_external_data=True)
        algo_config = KQuantWeightOnlyQuantConfig()
        quantizer = MatMulNBitsQuantizer(
            model=src_model,
            block_size=config.block_size,
            is_symmetric=config.is_symmetric,
            accuracy_level=config.accuracy_level,
            algo_config=algo_config,
        )
        quantizer.process()

        quantizer.model.save_model_to_file(str(output_model_path), use_external_data_format=True)

        return ONNXModelHandler(model_path=output_model_path)
