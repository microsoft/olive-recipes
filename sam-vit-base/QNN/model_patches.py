# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Optional

import torch
from torch import nn


class Conv2DInplaceLinear(nn.Module):
    """An implementation of Linear / Conv1D that uses a 1x1 Conv2D op instead.

    The Conv2D implementation for Qualcomm DSPs is faster than the Linear/Conv1D implementation.
    """

    @staticmethod
    # def from_linear(mod: torch.nn.Linear | torch.nn.Conv1d) -> Conv2DInplaceLinear:
    def from_linear(mod: torch.nn.Linear | torch.nn.Conv1d):
        weight: torch.Tensor | torch.nn.Parameter
        bias: torch.Tensor | torch.nn.Parameter | None
        if isinstance(mod, torch.nn.Linear):
            weight, bias = mod.weight, mod.bias
            bias = mod.bias
        elif isinstance(mod, torch.nn.Conv1d):
            weight, bias = mod.weight.T, mod.bias
        else:
            raise NotImplementedError

        out_features, in_features = weight.shape
        linear = Conv2DInplaceLinear(
            in_features,
            out_features,
            bias is not None,
            mod.device if hasattr(mod, "device") else None,
        )
        linear.conv2d.weight.data.copy_(weight.data[:, :, None, None])
        if bias is not None:
            assert linear.conv2d.bias is not None
            linear.conv2d.bias.data.copy_(bias.data)

        return linear

    def __init__(
        self,
        in_features,
        out_features,
        has_bias: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_features, out_features, 1, bias=has_bias)
        if device:
            self.conv2d.to(device)

    def __getattr__(self, attr):
        conv2d = self._modules["conv2d"]
        if attr == "conv2d":
            return conv2d
        return getattr(conv2d, attr)

    def forward(self, x: torch.Tensor):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(1)
        elif ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            pass

        x = x.permute(0, 3, 1, 2)  # (B, L, D) -> (B, D, 1, L)
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1)

        if ndim == 2:
            x = x.squeeze(1).squeeze(0)
        elif ndim == 3:
            x = x.squeeze(1)
        elif ndim == 4:
            pass
        return x


class SplitHeadSamVisionSdpaAttention(nn.Module):
    def __init__(self, attention_block):
        super().__init__()
        self.out_feature, self.in_feature = (
            attention_block.qkv.weight.shape[0] // 3,
            attention_block.qkv.weight.shape[1],
        )

        bias = attention_block.qkv.bias[: self.out_feature] is not None
        self.q = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.k = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.v = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.proj = Conv2DInplaceLinear.from_linear(attention_block.proj)
        self.num_heads = attention_block.num_attention_heads
        self.use_rel_pos = attention_block.use_rel_pos
        self.model = attention_block

        for chunk, proj_list in enumerate([self.q, self.k, self.v]):
            proj_list.conv2d.weight.data.copy_(
                attention_block.qkv.weight[
                    (chunk) * self.out_feature : (chunk + 1) * self.out_feature,
                    :,
                    None,
                    None,
                ]
            )

            assert proj_list.conv2d.bias is not None
            proj_list.conv2d.bias.data.copy_(
                attention_block.qkv.bias[(chunk) * self.out_feature : (chunk + 1) * self.out_feature,]
            )

    def get_decomposed_rel_pos(
        self,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: tuple[int, int],
        k_size: tuple[int, int],
    ) -> torch.Tensor:
        """Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.

        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            decomposed_rel_pos (`torch.Tensor`):
                decomposed relative position embeddings.

        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.model.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.model.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)

        # Original
        # rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        # rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)

        # Using MatMul
        rel_h = reshaped_query @ relative_position_height.transpose(1, 2)
        rel_w = (reshaped_query.transpose(1, 2) @ relative_position_width.transpose(1, 2)).transpose(1, 2)

        return rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]

    def forward(self, hidden_states: torch.Tensor, output_attentions=None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, height, width, _ = hidden_states.shape

        key = (
            self.k(hidden_states)
            .reshape(batch_size, height * width, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, height * width, -1)
        )
        value = (
            self.v(hidden_states)
            .reshape(batch_size, height * width, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, height * width, -1)
        )
        query = (
            self.q(hidden_states)
            .reshape(batch_size, height * width, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, height * width, -1)
        )

        attn_weights = (query * self.model.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            decomposed_rel_pos = self.get_decomposed_rel_pos(
                query,
                self.model.rel_pos_h,
                self.model.rel_pos_w,
                (height, width),
                (height, width),
            )
            decomposed_rel_pos = decomposed_rel_pos.reshape_as(attn_weights)
            attn_weights = attn_weights + decomposed_rel_pos

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        attn_probs = nn.functional.dropout(attn_weights, p=self.model.dropout, training=self.model.training)

        attn_output = (attn_probs @ value).reshape(batch_size, self.model.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.model.proj(attn_output)
        return attn_output, attn_weights


class Conv2DInplaceLinearSAMMLPBlock(nn.Module):
    """SAM MLPBlock that uses 1x1 Conv2D in place of linear layers."""

    def __init__(self, mlp_block) -> None:
        super().__init__()
        self.lin1 = Conv2DInplaceLinear.from_linear(mlp_block.lin1)
        self.lin2 = Conv2DInplaceLinear.from_linear(mlp_block.lin2)
        self.act = mlp_block.act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(hidden_states)))


class ModSamVisionLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attn = SplitHeadSamVisionSdpaAttention(self.model.attn)
        self.mlp = Conv2DInplaceLinearSAMMLPBlock(self.model.mlp)

    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
        batch_size, height, width, channel = hidden_states.shape

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size

        c1 = torch.zeros((batch_size, pad_h, width, channel))
        c2 = torch.zeros((batch_size, height + pad_h, pad_w, channel))

        hidden_states = torch.concatenate((hidden_states, c1), axis=1)
        hidden_states = torch.concatenate((hidden_states, c2), axis=2)

        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            pad_height // window_size,
            window_size,
            pad_width // window_size,
            window_size,
            channel,
        )
        windows = hidden_states.permute(0, 2, 1, 3, 4).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self,
        windows: torch.Tensor,
        window_size: int,
        padding_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> torch.Tensor:
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            pad_height // window_size,
            pad_width // window_size,
            window_size,
            window_size,
            -1,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous().reshape(batch_size, pad_height, pad_width, -1)

        return hidden_states[:, :height, :width, :].contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.model.layer_norm1(hidden_states)
        # Window partition
        if self.model.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.model.window_size)

        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
        )
        # Reverse window partition
        if self.model.window_size > 0:
            hidden_states = self.window_unpartition(
                hidden_states, self.model.window_size, padding_shape, (height, width)
            )

        hidden_states = residual + hidden_states
        layernorm_output = self.model.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)
        return hidden_states


class ModSamModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for i in range(len(self.model.vision_encoder.layers)):
            self.model.vision_encoder.layers[i] = ModSamVisionLayer(self.model.vision_encoder.layers[i])

    def forward(self, pixel_values, input_points):
        return self.model(pixel_values=pixel_values, input_points=input_points)


class ModSamVisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_encoder = ModSamModel(model).model.vision_encoder

    def forward(self, pixel_values):
        return self.vision_encoder(pixel_values=pixel_values)


class ModSamPromptEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        print(points.shape, labels.shape, pad)
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = (self.model.input_image_size, self.model.input_image_size)
        point_embedding = self.model.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(labels[..., None] == -1, self.model.not_a_point_embed.weight, point_embedding)

        # This is required for the ONNX export. The dtype, device need to be explicitly
        # specified as otherwise torch.onnx.export interprets as double
        point_embedding = torch.where(labels[..., None] != -10, point_embedding, torch.zeros_like(point_embedding))

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.model.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.model.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 2)[:, :, :, None],
            point_embedding + self.model.point_embed[2].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 3)[:, :, :, None],
            point_embedding + self.model.point_embed[3].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def forward(
        self,
        input_points: Optional[tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1
        if input_points is not None:
            batch_size = input_points.shape[0]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self.model._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.model.mask_embed(input_masks)
        else:
            dense_embeddings = self.model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.model.image_embedding_size[0], self.model.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class ModSamMaskdecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.prompt_encoder = ModSamPromptEncoder(self.model.prompt_encoder)

    def forward(self, input_points, input_labels, image_embeddings):
        return self.model(input_points=input_points, input_labels = input_labels, image_embeddings=image_embeddings)


class ModSamMaskPointDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_points, image_embeddings):
        return self.model(input_points=input_points, image_embeddings=image_embeddings)


class ModSamMaskBoxDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_boxes, image_embeddings):
        return self.model(input_boxes=input_boxes, image_embeddings=image_embeddings)
