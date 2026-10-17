"""Experimental Olive adapter for the pinned Nemotron-H target/MTP Mobius task."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from olive.model import HfModelHandler, ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
MODEL_REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
MOBIUS_REVISION = "e3d01a7712f38deb962e7107752da7fb668a7054"

logger = logging.getLogger(__name__)


class NemotronLightningBuilder(Pass):
    """Export both BF16 graphs without implying parity or a speculative runtime."""

    @classmethod
    def is_accelerator_agnostic(cls, accelerator_spec):
        return False

    @classmethod
    def _default_config(cls, accelerator_spec):
        return {
            "precision": PassConfigParam(type_=Literal["bf16"], default_value="bf16"),
            "allow_unvalidated_export": PassConfigParam(type_=bool, default_value=False),
        }

    def _run_for_config(self, model, config, output_model_path):
        if not config.allow_unvalidated_export:
            raise ValueError(
                "Full-model parity is blocked in Mobius PR #761. "
                "Explicit allow_unvalidated_export=True is required for development exports."
            )
        if (
            self.accelerator_spec.execution_provider != "CUDAExecutionProvider"
            or self.accelerator_spec.accelerator_type != "gpu"
        ):
            raise ValueError("This recipe requires GPU/CUDAExecutionProvider; no provider fallback is allowed.")
        if config.precision != "bf16":
            raise ValueError("This recipe requires explicit BF16 precision.")
        if not isinstance(model, HfModelHandler):
            raise ValueError("NemotronLightningBuilder requires an HfModel input.")
        load_kwargs = model.get_load_kwargs()
        if model.model_name_or_path != MODEL_ID or load_kwargs.get("revision") != MODEL_REVISION:
            raise ValueError("This recipe requires its exact pinned Lightning BF16 model and revision.")
        if load_kwargs.get("trust_remote_code", False):
            raise ValueError("Remote model code is not permitted by this recipe.")
        output_dir = Path(output_model_path)
        if output_dir.is_symlink() or (
            output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))
        ):
            raise FileExistsError("Use a fresh, empty export directory; existing artifacts must not be overwritten.")

        from mobius import build

        logger.warning("Experimental target/MTP CUDA graph export; full-model parity and runtime remain unvalidated.")
        package = build(
            MODEL_ID,
            revision=MODEL_REVISION,
            task="nemotron-h-mtp",
            dtype="bf16",
            execution_provider="cuda",
            load_weights=True,
            trust_remote_code=False,
        )
        if set(package) != {"decoder", "mtp"}:
            raise RuntimeError("The pinned Mobius task must return both decoder and mtp components.")
        package.save(str(output_dir), external_data="onnx")
        components = []
        for name in ("decoder", "mtp"):
            directory = output_dir / name
            if not (directory / "model.onnx").is_file():
                raise RuntimeError(f"Missing exported {name}/model.onnx; the paired export is incomplete.")
            components.append(
                ONNXModelHandler(
                    model_path=str(directory),
                    onnx_file_name="model.onnx",
                    model_attributes={"mobius_component": name},
                )
            )
        return CompositeModelHandler(
            model_components=components,
            model_component_names=["decoder", "mtp"],
            model_path=str(output_dir),
            model_attributes={
                "no_flatten": True,
                "additional_files": sorted(str(path) for path in output_dir.iterdir() if path.is_file()),
                "mobius_package_keys": ["decoder", "mtp"],
                "mobius_revision": MOBIUS_REVISION,
                "model_revision": MODEL_REVISION,
                "validation_status": "full_model_parity_blocked",
                "speculative_runtime": "not_implemented",
            },
        )
