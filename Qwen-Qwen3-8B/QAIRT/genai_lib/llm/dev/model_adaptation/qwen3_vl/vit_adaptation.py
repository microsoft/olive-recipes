#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
Model adaptations required for Qwen 3VL ViT, based on transformers v4.57.1, https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
"""

import torch
import torch.nn.functional as F
from typing import Optional

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionConfig,
    Qwen3VLVisionModel,
    rotate_half
)


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



def qc_apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Qc Adapted function based on `modeling_qwen3_vl.apply_rotary_pos_emb_vision`
    If the sinusoids have the same shape as the query/key vectors in the last dimension, maintain HF compatibility
    If the sinusoids are only half the size of the query/key vectors, it means that we are on the adapted route
    In the adapted route we use `apply_rope_single` to apply rotary embedding on query and key vectors separately
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    ########### Qc Adaptation ###########
    if cos.shape[-1] == q.shape[-1]:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        q_embed = _apply_rope_single(q, (cos, sin))
        k_embed = _apply_rope_single(k, (cos, sin))
    #####################################
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed



class QcQwen3VLVisionModel(Qwen3VLVisionModel):
    def __init__(self, config: Qwen3VLVisionConfig, *inputs, **kwargs) -> None:
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

        fixed_grid_thw = torch.tensor(fixed_grid_thw, dtype = torch.int64).unsqueeze(0)
        cu_seqlens = torch.repeat_interleave(fixed_grid_thw[:, 1] * fixed_grid_thw[:, 2], fixed_grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=fixed_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent = False)

        # Keep the grid around; you can also register it as a buffer if it's constant
        self.register_buffer("fixed_grid_thw", fixed_grid_thw, persistent=False)
        # As the pos_embeds depend on "pos_embed.weight" to be initialized,
        # we make a placeholder buffer; and it will be lazily initialized during the first `forward()`.
        self.register_buffer("pos_embeds", None, persistent=False)
        self._is_pos_embeds_initialized = False


    @torch.no_grad()
    def _build_pos_embeds(self):
        # If params are still on meta, we can’t build yet
        if self.pos_embed.weight.device.type == "meta":
            return

        device = self.pos_embed.weight.device
        grid = self.fixed_grid_thw.to(device)

        pe = self.fast_pos_embed_interpolate(grid)
        # Ensure no autograd graph is kept
        pe = pe.detach()

        # Re-register the buffer on the correct device/dtype
        self.register_buffer("pos_embeds", pe, persistent=False)
        self._is_pos_embeds_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_cos: Optional[torch.Tensor]=None,
        position_embeddings_sin: Optional[torch.Tensor]=None,
        grid_thw: Optional[torch.Tensor]=None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        # Lazily initialize pos_embeds on first forward, which won't be traced by jit,
        # as we'll call the adapted_vit.forward() before prepare_model
        if not self._is_pos_embeds_initialized:
            self._build_pos_embeds()

        if self.pos_embeds is None:
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        else:
            pos_embeds = self.pos_embeds
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds
        if position_embeddings_cos is not None and position_embeddings_sin is not None and grid_thw is None:
            position_embeddings = (position_embeddings_cos, position_embeddings_sin)
        elif position_embeddings_cos is None and position_embeddings_sin is None and grid_thw is not None:
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
        else:
            raise ValueError(
                "Either `position_embeddings_cos` and `position_embeddings_sin`, "
                "or `grid_thw` must be provided."
                "Either got both inputs, or none."
            )

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        return hidden_states, tuple(deepstack_feature_lists)
