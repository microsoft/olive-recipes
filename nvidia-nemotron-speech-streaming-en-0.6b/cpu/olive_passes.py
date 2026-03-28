# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Custom Olive passes for the Nemotron Speech Streaming recipe.

NemotronExport:
    Wraps scripts/export_nemotron_to_onnx_static_shape.py so that the NeMo
    model export step can be executed as the first pass in an Olive workflow
    run, feeding directly into the graph-fusion and INT4-quantization passes.

NemotronKQuantQuantization:
    Applies INT4 weight-only k-quant quantization to an ONNX encoder via
    OnnxRuntime's MatMulNBitsQuantizer with KQuantWeightOnlyQuantConfig.
    After quantizing the encoder, copies all supporting files (decoder.onnx,
    joint.onnx, config files, tokenizer files) from the export directory and
    downloads silero_vad.onnx into the final output directory so that all
    artifacts needed for ORT GenAI are co-located.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import onnx
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam

# Populated by NemotronExport._run_for_config so that NemotronKQuantQuantization
# can locate the supporting files (decoder, joint, configs, tokenizer) that were
# produced alongside the encoder during the export pass.
_export_dir: Optional[Path] = None


class NemotronExport(Pass):
    """Export the Nemotron Speech Streaming NeMo model to ONNX.

    Runs scripts/export_nemotron_to_onnx_static_shape.py via subprocess and
    returns an ONNXModelHandler pointing to the exported encoder so that
    subsequent Olive passes (OrtTransformersOptimization, NemotronKQuantQuantization)
    can operate on it.

    The decoder, joint network, and config files produced alongside the encoder
    are written to the same Olive-managed output directory. The export directory
    path is recorded in _export_dir so that NemotronKQuantQuantization can copy
    these supporting files to the final output directory after quantization.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict:
        return {
            "model_name": PassConfigParam(
                type_=str,
                default_value="nvidia/nemotron-speech-streaming-en-0.6b",
                description=(
                    "HuggingFace model ID or path to a local .nemo file to export. "
                    "Passed directly to the export script's --model_name argument."
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
        # Resolve the output path to a concrete encoder.onnx file path
        # so it matches the filename the export script produces.
        output_model_path = resolve_onnx_path(output_model_path, "encoder.onnx")
        output_dir = Path(output_model_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # The export script lives at scripts/ one level above cpu/
        script = Path(__file__).parent.parent / "scripts" / "export_nemotron_to_onnx_static_shape.py"

        # Resolve the device value – Olive's resource-path scanner resolves bare
        # strings that match local directory names (e.g. "cpu" → the ./cpu/ folder).
        # Retrieve the original string from the default config if the value was
        # coerced into an absolute path.
        device = config.device
        if Path(str(device)).is_absolute():
            device = Path(str(device)).name

        cmd = [
            sys.executable, str(script),
            "--model_name", config.model_name,
            "--output_dir", str(output_dir),
            "--chunk_size", str(config.chunk_size),
            "--left_chunks", str(config.left_chunks),
            "--device", device,
        ]
        if config.streaming:
            cmd.append("--streaming")

        subprocess.run(cmd, check=True)

        # Record the export directory so NemotronKQuantQuantization can find
        # the supporting files (decoder, joint, configs, tokenizer) that were
        # written alongside the encoder.
        global _export_dir
        _export_dir = output_dir

        return ONNXModelHandler(model_path=output_model_path)


class NemotronKQuantQuantization(Pass):
    """Apply INT4 k-quant weight-only quantization to an ONNX encoder.

    Uses OnnxRuntime's MatMulNBitsQuantizer with KQuantWeightOnlyQuantConfig
    to produce a MatMulNBits INT4-only ONNX model.  All MatMul weights are
    quantized; non-weight tensors remain FP32.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict:
        return {
            "block_size": PassConfigParam(
                type_=int,
                default_value=32,
                description=(
                    "Block size for k-quant quantization. "
                    "Smaller values improve accuracy at the cost of model size. Default: 32."
                ),
            ),
            "is_symmetric": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Use symmetric quantization. Default: True.",
            ),
            "accuracy_level": PassConfigParam(
                type_=int,
                default_value=4,
                description=(
                    "Accuracy level for the MatMulNBits computation kernel "
                    "(0–4). Higher values use more accurate accumulation. Default: 4."
                ),
            ),
        }

    def _run_for_config(self, model: ONNXModelHandler, config, output_model_path: str) -> ONNXModelHandler:
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            KQuantWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        output_dir = Path(output_model_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # MatMulNBitsQuantizer requires the full model graph in memory to rewrite
        # MatMul nodes to MatMulNBits.  For the Nemotron encoder (~600 MB), this
        # is manageable; external data is loaded so all weight tensors are present.
        m = onnx.load(model.model_path, load_external_data=True)
        quantizer = MatMulNBitsQuantizer(
            model=m,
            block_size=config.block_size,
            is_symmetric=config.is_symmetric,
            accuracy_level=config.accuracy_level,
            algo_config=KQuantWeightOnlyQuantConfig(),
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(output_model_path), use_external_data_format=True)

        # Copy decoder, joint, config, and tokenizer files from the export directory
        # to the final output directory so all ORT GenAI artifacts are co-located.
        if _export_dir is not None and _export_dir.is_dir():
            copied = 0
            for src_file in sorted(_export_dir.iterdir()):
                if not src_file.is_file():
                    continue
                if src_file.name.startswith("encoder"):
                    continue  # encoder is produced by this pass; skip
                dst_file = output_dir / src_file.name
                if src_file.resolve() != dst_file.resolve():
                    shutil.copy2(str(src_file), str(dst_file))
                    copied += 1
            print(f"  Copied {copied} supporting files to: {output_dir}")
        else:
            print("  Warning: export directory not found; supporting files not copied.")

        # Download the Silero VAD ONNX model alongside the other ONNX models.
        try:
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(
                repo_id="onnx-community/silero-vad",
                filename="onnx/model.onnx",
            )
            dst = output_dir / "silero_vad.onnx"
            shutil.copy2(cached, str(dst))
            print(f"  Saved Silero VAD model to: {dst}")
        except Exception as exc:
            print(
                f"  Warning: Silero VAD download failed ({exc}).\n"
                "  You can download it manually with:\n"
                "    huggingface-cli download onnx-community/silero-vad "
                f"--include onnx/model.onnx --local-dir .\n"
                f"  and copy onnx/model.onnx to {output_dir / 'silero_vad.onnx'}"
            )

        return ONNXModelHandler(model_path=output_model_path)
