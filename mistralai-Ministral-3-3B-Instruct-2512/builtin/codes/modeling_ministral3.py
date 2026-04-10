# Copyright 2025 HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Adapted from transformers/models/mistral3/modeling_mistral3.py
#
# REFERENCE ONLY: This module is NOT used by optimize.py (which uses mobius
# for vision/embedding export). It is kept as a reference implementation
# showing how to build an ONNX-export-friendly Ministral3 vision + embedding
# model for potential future Olive-based export.

from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoModel
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config


class Mistral3PatchMerger(nn.Module):
    """ONNX-export-friendly Mistral3PatchMerger.

    Uses pure tensor operations during export instead of Python for-loops.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def forward(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor:
        if torch.compiler.is_exporting():
            return self._forward_export(image_features, image_sizes)
        return self._forward_eager(image_features, image_sizes)

    def _forward_export(self, image_features, image_sizes):
        patch_h = image_sizes[0, 0] // self.patch_size
        patch_w = image_sizes[0, 1] // self.patch_size
        d = image_features.shape[-1]

        image_grid = (
            image_features.view(patch_h, patch_w, d).permute(2, 0, 1).unsqueeze(0)
        )

        torch._check(image_grid.shape[2] != 0)
        torch._check(image_grid.shape[3] != 0)
        torch._check(image_grid.shape[2] // self.spatial_merge_size > 0)
        torch._check(image_grid.shape[3] // self.spatial_merge_size > 0)

        grid = torch.nn.functional.unfold(
            image_grid,
            kernel_size=self.spatial_merge_size,
            stride=self.spatial_merge_size,
        )
        image_features = grid.view(d * self.spatial_merge_size**2, -1).t()
        return self.merging_layer(image_features)

    def _forward_eager(self, image_features, image_sizes):
        image_sizes_list = [
            (sz[0] // self.patch_size, sz[1] // self.patch_size) for sz in image_sizes
        ]
        tokens_per_image = [h * w for h, w in image_sizes_list]
        d = image_features.shape[-1]

        permuted = []
        for idx, image_tokens in enumerate(image_features.split(tokens_per_image)):
            h, w = image_sizes_list[idx]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            permuted.append(grid.view(d * self.spatial_merge_size**2, -1).t())

        return self.merging_layer(torch.cat(permuted, dim=0))


def pixtral_vision_forward_export(self, pixel_values, **kwargs):
    """ONNX-export-friendly forward for PixtralVisionModel (batch=1).

    Skips generate_block_attention_mask and computes position_ids inline.
    """
    torch._check(pixel_values.shape[0] == 1)

    target_dtype = self.patch_conv.weight.dtype
    patch_embeds = self.patch_conv(pixel_values.to(dtype=target_dtype))

    grid_h = patch_embeds.shape[2]
    grid_w = patch_embeds.shape[3]

    patch_embeds = patch_embeds[0].flatten(1).T.unsqueeze(0)
    patch_embeds = self.ln_pre(patch_embeds)

    max_width = self.config.image_size // self.config.patch_size
    h_indices = torch.arange(grid_h, device=pixel_values.device)
    w_indices = torch.arange(grid_w, device=pixel_values.device)
    mesh_h, mesh_w = torch.meshgrid(h_indices, w_indices, indexing="ij")
    position_ids = (mesh_h * max_width + mesh_w).reshape(-1)
    kwargs["position_ids"] = position_ids.unsqueeze(0)

    position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

    return self.transformer(
        patch_embeds,
        attention_mask=None,
        position_embeddings=position_embeddings,
        **kwargs,
    )


def _pixtral_vision_forward_dispatch(self, pixel_values, **kwargs):
    if torch.compiler.is_exporting():
        return pixtral_vision_forward_export(self, pixel_values, **kwargs)
    return self._original_forward(pixel_values, **kwargs)


def patch_model_for_onnx_export(model):
    """Apply ONNX-export-friendly patches to a Mistral 3 model."""
    import types

    if hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
        patch_merger = model.model.multi_modal_projector.patch_merger
        vision_tower = model.model.vision_tower
    elif hasattr(model, "multi_modal_projector"):
        patch_merger = model.multi_modal_projector.patch_merger
        vision_tower = model.vision_tower
    else:
        raise ValueError("Cannot find multi_modal_projector.patch_merger on the model.")

    patch_merger.__class__ = Mistral3PatchMerger

    vision_tower._original_forward = vision_tower.forward
    vision_tower.forward = types.MethodType(
        _pixtral_vision_forward_dispatch, vision_tower
    )

    return model


class Ministral3Model(nn.Module):
    """Ministral3 composite model for vision + embedding ONNX export.

    Wraps HF Mistral3Model and provides:
    - get_image_features(): vision encoder export
    - get_fused_input_embeddings(): embedding fusion export
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        # Build the full HF model, then patch for export
        self.hf_model = AutoModel.from_config(
            config, attn_implementation="sdpa", trust_remote_code=True
        )
        patch_model_for_onnx_export(self.hf_model)

        # Expose sub-components for weight loading
        self.vision_tower = self.hf_model.vision_tower
        self.multi_modal_projector = self.hf_model.multi_modal_projector
        self.embed_tokens = self.hf_model.language_model.embed_tokens

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Vision encoder: pixel_values -> image_features."""
        image_outputs = self.vision_tower(pixel_values, return_dict=True)
        selected_image_feature = image_outputs.last_hidden_state

        image_sizes = torch.tensor(
            [[pixel_values.shape[-2], pixel_values.shape[-1]]],
            dtype=torch.int64,
            device=pixel_values.device,
        )
        image_features = self.multi_modal_projector(
            selected_image_feature.squeeze(0), image_sizes
        )
        return image_features

    def get_fused_input_embeddings(
        self, input_ids: torch.LongTensor, image_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embedding fusion: input_ids + image_features -> inputs_embeds."""
        inputs_embeds = self.embed_tokens(input_ids)
        if image_features is not None:
            image_features = image_features.to(inputs_embeds.dtype)
            special_image_mask = input_ids == self.config.image_token_index
            expanded_mask = (
                special_image_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, image_features)
        return inputs_embeds

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use get_image_features() or get_fused_input_embeddings() via method swap."
        )


__all__ = ["Ministral3Model", "patch_model_for_onnx_export"]
