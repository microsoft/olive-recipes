# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from olive.passes.onnx.model_builder import ModelBuilder


class LingModelBuilder(ModelBuilder):
    """Use GenAI's native Bailing checkpoint and embedding exporters."""

    def maybe_patch_quant(self):
        from onnxruntime_genai.models import builder

        # Olive's patches target Olive-quantized checkpoints, not the native Ling loader.
        if not hasattr(builder.Model, "make_embedding_lookup"):
            raise RuntimeError("Ling recipes require the native embedding exporter from onnxruntime-genai PR #2843")
        if builder.Model.make_embedding.__module__.startswith("olive."):
            raise RuntimeError(
                "GenAI's embedding exporter was already patched by Olive; run the Ling recipe in a fresh process"
            )
