#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


'''
Model adaptations required for Qwen 2.5VL ViT
'''


from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention, Qwen2_5_VisionTransformerPretrainedModel, repeat_kv, apply_rotary_pos_emb_vision
from genai_lib.llm.dev.model_adaptation.qwen2_5_vl.vit_utils import vit_prepare_attention_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _apply_rope_single(x, rope_vals: tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:,:,:x.shape[-1]//2] # extract first half elements
    x_im = x[:,:,x.shape[-1]//2:] # extract second half elements

    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real,x_prod_im),dim=2).view(*x.shape)
    return x



class QcQwen2_5_VLVisionAttention(Qwen2_5_VLVisionAttention):

    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        # The adapted attention class forward is a combination of `Qwen2_5_VLVisionAttention` and `eager_attention_forward` from modeling_qwen2_5_vl

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        if position_embeddings is None:
            cos = rotary_pos_emb.cos()
            sin = rotary_pos_emb.sin()
        else:
            # QC Adaptation: Receive precomputed rope sin/cos as inputs
            cos, sin = position_embeddings

        orig_q_dtype = query_states.dtype
        orig_k_dtype = key_states.dtype
        if cos.shape[-1] == query_states.shape[-1]:
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        else:
            # QC Adaptation: apply RoPE separately to query and key
            query_states, key_states = query_states.float(), key_states.float()
            cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
            query_states = _apply_rope_single(query_states, (cos, sin))
            key_states = _apply_rope_single(key_states, (cos, sin))
        query_states = query_states.to(orig_q_dtype)
        key_states = key_states.to(orig_k_dtype)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            if attention_mask.shape[-1] != value_states.shape[-2]:
                attention_mask = attention_mask[:, :, :, : value_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output



class QcQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        # Expectation: a 3-list [T, H, W] set by the caller (e.g., via processor output) since this class is specifically designed for fixed size images.
        if not hasattr(config, "fixed_grid_thw"):
            raise ValueError(
                "Missing required config: `fixed_grid_thw`.\n"
                "This adapted ViT implementation is based on the assumption of a constant size image input."
                "It expects a constant image grid (T, H, W) through the `config` at init."
                "If you are using a vision-only config, set `config.fixed_grid_thw = [T, H, W]`.\n"
                "If you are at the LLM level (e.g., `Qwen2_5_VLConfig`), set "
                "`config.vision_config.fixed_grid_thw = [T, H, W]` "
                "(e.g., from `processor.image_processor(...)[\"image_grid_thw\"].squeeze(0).tolist()`)."
            )
        fixed_grid_thw = config.fixed_grid_thw
        if not isinstance(fixed_grid_thw, (list, tuple)) or len(fixed_grid_thw) != 3:
            raise ValueError(
                f"Invalid `fixed_grid_thw`: expected a sequence of length 3 [T, H, W], "
                f"got {fixed_grid_thw!r}."
            )

        seq_len = fixed_grid_thw[1] * fixed_grid_thw[2]
        fixed_grid_thw = torch.tensor(fixed_grid_thw, dtype = torch.int64).unsqueeze(0)
        window_index, cu_window_seqlens = self.get_window_index(fixed_grid_thw)

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            dtype=fixed_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        cu_seqlens = torch.repeat_interleave(fixed_grid_thw[:, 1] * fixed_grid_thw[:, 2], fixed_grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=fixed_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = torch.empty((seq_len,))
        window_attention_mask = vit_prepare_attention_mask(hidden_states, cu_window_seqlens)
        full_attention_mask = vit_prepare_attention_mask(hidden_states, cu_seqlens)

        self.register_buffer(name = "window_index", tensor = window_index, persistent = False)
        self.register_buffer(name = "cu_window_seqlens", tensor = cu_window_seqlens, persistent = False)
        self.register_buffer(name = "cu_seqlens", tensor = cu_seqlens, persistent = False)
        self.register_buffer(name = "full_attention_mask", tensor = full_attention_mask, persistent = False)
        self.register_buffer(name = "window_attention_mask", tensor = window_attention_mask, persistent = False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings_cos: Optional[torch.Tensor] = None,
            position_embeddings_sin: Optional[torch.Tensor] = None,
            grid_thw: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.patch_embed(hidden_states)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[self.window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        if (grid_thw is not None) and (position_embeddings_cos is None and position_embeddings_sin is None):
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            rotary_pos_emb = rotary_pos_emb[self.window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

        elif (position_embeddings_cos is not None and position_embeddings_sin is not None) and (grid_thw is None):
            # QC Adaptation: Position Embeddings - Difference wrt `Qwen2_5_VisionTransformerPretrainedModel.forward` in transformers==4.53.1
            # We calculate the .cos() and .sin() from rotary_pos_emb outside the graph since it is not accurate on NSP
            # However, we leave the part of code doing reshuffle using window index inside the graph, since window index is calculated inside the graph
            # `x` in the following code is analogous to `rotary_pos_emb` from original code. We need to do the permutation for cos as well as sin, hence the for loop
            # We also ommit the line https://github.com/huggingface/transformers/blob/896e9cea1ade521b2648f4798218550f6c72190c/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L472 since we are using the apply_rope_single function in the attention class
            position_embeddings = []
            for x in position_embeddings_cos, position_embeddings_sin:
                x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
                x = x[self.window_index, :, :]
                x = x.reshape(seq_len, -1)
                position_embeddings.append(x)
            position_embeddings = tuple(position_embeddings)

        else:
            raise ValueError(
                "Position embedding inputs are mutually exclusive. "
                "Provide either `grid_thw` (for original HF forward) "
                "OR `position_embeddings_cos` and `position_embeddings_sin` (for adapted forward), but not both."
    )

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask = self.full_attention_mask
            else:
                attention_mask = self.window_attention_mask

            hidden_states = blk(
                hidden_states,
                # We do not need to use `cu_seqlens` in the attention operation since we are using eager attention. We pass a placeholder tensor since the `Qwen2_5_VLVisionBlock` class expects this argument
                cu_seqlens=torch.tensor([]),
                rotary_pos_emb=None,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(self.window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states
