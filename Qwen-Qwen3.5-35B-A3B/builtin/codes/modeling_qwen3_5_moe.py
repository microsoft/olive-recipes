# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Self-contained ONNX-export-friendly Qwen3.5-MoE model.
# Vision encoder is identical to non-MoE Qwen3.5. MoE text components are
# included for completeness but the text decoder is exported via OGA ModelBuilder.
# Used by user_script.py for vision and embedding ONNX export.
# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import auto_docstring, logging

try:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeVisionConfig,
        Qwen3_5MoeTextConfig,
    )
except ImportError:
    from transformers import AutoConfig
    Qwen3_5MoeConfig = AutoConfig
    Qwen3_5MoeVisionConfig = None
    Qwen3_5MoeTextConfig = None

logger = logging.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Vision encoder components (identical to non-MoE Qwen3.5)
# ═══════════════════════════════════════════════════════════════


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, seqlen) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim).reshape(
        batch, num_kv_heads * n_rep, slen, head_dim
    )


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class VisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if getattr(torch.compiler, "is_exporting", lambda: False)():
            attn_output = torch.onnx.ops.symbolic(
                "custom::PackedAttention",
                (q, k, v, cu_seqlens),
                dict(scale=self.scaling, num_heads=self.num_heads),
                dtype=q.dtype, shape=(q.shape[0], q.shape[2], q.shape[1], q.shape[3]), version=1,
            ).to(self.proj.weight.device)
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]
            attn_outputs = []
            for qi, ki, vi in zip(*splits):
                out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=None, dropout_p=0.0, scale=self.scaling, is_causal=False)
                attn_outputs.append(out.transpose(1, 2))
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class VisionBlock(GradientCheckpointingLayer):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config=config)
        self.mlp = VisionMLP(config=config)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens,
                                                   rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings, **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ═══════════════════════════════════════════════════════════════
# Vision model
# ═══════════════════════════════════════════════════════════════


@auto_docstring
class Qwen3_5MoePreTrainedModel(PreTrainedModel):
    config_class = Qwen3_5MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True
    _supports_attention_backend = True


class Qwen3_5MoeVisionModel(Qwen3_5MoePreTrainedModel):
    _no_split_modules = ["VisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = VisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config=config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [VisionPatchMerger(config=config, use_postshuffle_norm=True) for _ in range(len(config.deepstack_visual_indexes))]
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        merge_size = self.spatial_merge_size
        max_hw = grid_thw[:, 1:].max()
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        all_embeddings = []
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            torch._check(merged_h.item() >= 1)
            torch._check(merged_w.item() >= 1)
            torch._check(num_frames.item() >= 1)

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]).expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = (block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]).expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            coords = coords.repeat(num_frames, 1)
            all_embeddings.append(freq_table[coords].flatten(1))

        return torch.cat(all_embeddings, dim=0)

    def fast_pos_embed_interpolate(self, grid_thw):
        merge_size = self.config.spatial_merge_size
        dev = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        n = self.num_grid_per_side

        all_pos_embeds = []
        for t, h, w in zip(grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]):
            torch._check(t.item() >= 1)
            torch._check(h.item() >= 2)
            torch._check(w.item() >= 2)

            h_idxs = torch.arange(h, dtype=torch.float32, device=dev) * ((n - 1) / (h - 1))
            w_idxs = torch.arange(w, dtype=torch.float32, device=dev) * ((n - 1) / (w - 1))

            h_floor, w_floor = h_idxs.int(), w_idxs.int()
            h_ceil = (h_floor + 1).clamp(max=n - 1)
            w_ceil = (w_floor + 1).clamp(max=n - 1)
            dh = (h_idxs - h_floor.float()).to(dtype)
            dw = (w_idxs - w_floor.float()).to(dtype)

            base_h = h_floor.long() * n
            base_hc = h_ceil.long() * n
            idx_00 = (base_h[:, None] + w_floor.long()[None]).reshape(-1)
            idx_01 = (base_h[:, None] + w_ceil.long()[None]).reshape(-1)
            idx_10 = (base_hc[:, None] + w_floor.long()[None]).reshape(-1)
            idx_11 = (base_hc[:, None] + w_ceil.long()[None]).reshape(-1)

            wt_00 = ((1.0 - dh)[:, None] * (1.0 - dw)[None]).reshape(-1)
            wt_01 = ((1.0 - dh)[:, None] * dw[None]).reshape(-1)
            wt_10 = (dh[:, None] * (1.0 - dw)[None]).reshape(-1)
            wt_11 = (dh[:, None] * dw[None]).reshape(-1)

            pos = (
                self.pos_embed(idx_00.to(dev)) * wt_00[:, None]
                + self.pos_embed(idx_01.to(dev)) * wt_01[:, None]
                + self.pos_embed(idx_10.to(dev)) * wt_10[:, None]
                + self.pos_embed(idx_11.to(dev)) * wt_11[:, None]
            )
            pos = pos.repeat(t, 1)
            pos = pos.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            all_pos_embeds.append(pos)

        return torch.cat(all_pos_embeds)

    def forward(self, hidden_states, grid_thw, **kwargs):
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs)

        hidden_states = self.merger(hidden_states)
        return hidden_states


# ═══════════════════════════════════════════════════════════════
# MoE text components (for completeness; text decoder exported via OGA)
# ═══════════════════════════════════════════════════════════════


class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (1 + self.weight) * hidden_states.to(input_dtype)


class Qwen3_5MoeMLP(nn.Module):
    """Dense SwiGLU MLP (used for shared expert)."""
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, num_experts, hidden_size, num_experts_per_tok):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_values, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        return router_logits, top_values.to(router_logits.dtype), top_indices


class Qwen3_5MoeExperts(nn.Module):
    """Packed expert weights: gate_up_proj [E, 2*inter, hidden], down_proj [E, hidden, inter]."""
    def __init__(self, num_experts, hidden_size, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

        for eidx in expert_hit:
            eidx = eidx[0]
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            current = hidden_states[token_idx]
            gate, up = F.linear(current, self.gate_up_proj[eidx]).chunk(2, dim=-1)
            out = F.linear(self.act_fn(gate) * up, self.down_proj[eidx])
            final.index_add_(0, token_idx, (out * top_k_weights[token_idx, top_k_pos, None]).to(final.dtype))
        return final


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Full MoE block: router + routed experts + shared expert + shared gate."""
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config.num_experts, config.hidden_size, config.num_experts_per_tok)
        self.experts = Qwen3_5MoeExperts(config.num_experts, config.hidden_size, config.moe_intermediate_size, config.hidden_act)
        self.shared_expert = Qwen3_5MoeMLP(config.hidden_size, config.shared_expert_intermediate_size, config.hidden_act)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        shared_out = self.shared_expert(flat)
        _, routing_weights, selected_experts = self.gate(flat)
        expert_out = self.experts(flat, selected_experts, routing_weights)
        shared_out = torch.sigmoid(self.shared_expert_gate(flat)) * shared_out
        return (expert_out + shared_out).reshape(batch_size, seq_len, hidden_dim)


# ═══════════════════════════════════════════════════════════════
# Top-level Qwen3.5-MoE model (vision + embedding shell)
# ═══════════════════════════════════════════════════════════════


class Qwen3_5MoeModel(Qwen3_5MoePreTrainedModel):
    """Qwen3.5-MoE composite model for vision + embedding ONNX export.

    The text decoder (with MoE + GatedDeltaNet) is exported via OGA ModelBuilder.
    This class provides:
    - get_image_features(): vision encoder export
    - get_fused_input_embeddings(): embedding fusion export
    """
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    _no_split_modules = ["VisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        text_config = config.text_config
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_image_features(self, pixel_values, image_grid_thw=None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        return torch.split(image_embeds, split_sizes)

    def get_fused_input_embeddings(self, input_ids, image_features=None):
        mask = input_ids == self.config.image_token_id
        safe_ids = torch.where(mask, torch.zeros_like(input_ids), input_ids)
        inputs_embeds = self.embed_tokens(safe_ids)
        if image_features is not None:
            mask_3d = mask.unsqueeze(-1).expand_as(inputs_embeds)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask_3d, image_features)
        return inputs_embeds

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3_5MoeModel.forward() should not be called directly. "
            "Use get_image_features() or get_fused_input_embeddings() via method swap."
        )


__all__ = [
    "Qwen3_5MoeModel",
    "Qwen3_5MoePreTrainedModel",
    "Qwen3_5MoeVisionModel",
    "Qwen3_5MoeSparseMoeBlock",
    "Qwen3_5MoeExperts",
    "Qwen3_5MoeTopKRouter",
    "Qwen3_5MoeMLP",
    "Qwen3_5MoeRMSNorm",
]
