#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

# =============================================================================
# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

""" This file provides adaptations to clip attention. These adaptations are being done to optimize the model execution on the HTP backend. https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py"""

import torch
from torch import nn
from aimet_torch import elementwise_ops
from typing import Optional, Tuple


class CLIPAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper
    Modified from source: https://github.com/huggingface/transformers/blob/5d29530ea25fab34eaf193116512753609f2ff54/src/transformers/models/clip/modeling_clip.py#L224
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = torch.nn.Dropout(config.attention_dropout)

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.bmm_1 = elementwise_ops.MatMul()
        self.bmm_2 = elementwise_ops.MatMul()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_causal_attn = elementwise_ops.Add()
        self.mask_attn = elementwise_ops.Add()

        self.is_sha = False
        self.mha_conv = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _split_attention_heads(self):
        self.q_convs = nn.ModuleList([nn.Conv2d(self.embed_dim, self.head_dim, 1, bias=self.q_proj.bias is not None) for _ in range(self.num_heads)])
        self.k_convs = nn.ModuleList([nn.Conv2d(self.embed_dim, self.head_dim, 1, bias=self.k_proj.bias is not None) for _ in range(self.num_heads)])
        self.v_convs = nn.ModuleList([nn.Conv2d(self.embed_dim, self.head_dim, 1, bias=self.v_proj.bias is not None) for _ in range(self.num_heads)])

        self.matmul_1 = nn.ModuleList([elementwise_ops.MatMul() for _ in range(self.num_heads)])
        self.matmul_2 = nn.ModuleList([elementwise_ops.MatMul() for _ in range(self.num_heads)])
        self.concat_1 = elementwise_ops.Concat(-1)

        for idx in range(self.num_heads):
            self.q_convs[idx].weight.data.copy_(self.q_proj.weight[idx*self.head_dim:(idx+1)*self.head_dim, :, None, None]).to(self.q_proj.weight)
            self.k_convs[idx].weight.data.copy_(self.k_proj.weight[idx*self.head_dim:(idx+1)*self.head_dim, :, None, None]).to(self.k_proj.weight)
            self.v_convs[idx].weight.data.copy_(self.v_proj.weight[idx*self.head_dim:(idx+1)*self.head_dim, :, None, None]).to(self.v_proj.weight)
            if self.q_convs[idx].bias is not None: self.q_convs[idx].bias.data.copy_(self.q_proj.bias[idx*self.head_dim:(idx+1)*self.head_dim]).to(self.q_proj.weight)
            if self.k_convs[idx].bias is not None: self.k_convs[idx].bias.data.copy_(self.k_proj.bias[idx*self.head_dim:(idx+1)*self.head_dim]).to(self.k_proj.weight)
            if self.v_convs[idx].bias is not None: self.v_convs[idx].bias.data.copy_(self.v_proj.bias[idx*self.head_dim:(idx+1)*self.head_dim]).to(self.v_proj.weight)
        self.is_sha = True

    def _replace_linear_to_conv(self):
        self.k_proj_conv = nn.Conv2d(self.embed_dim, self.embed_dim, (1, 1))
        self.v_proj_conv = nn.Conv2d(self.embed_dim, self.embed_dim, (1, 1))
        self.q_proj_conv = nn.Conv2d(self.embed_dim, self.embed_dim, (1, 1))
        self.out_proj_conv = nn.Conv2d(self.embed_dim, self.embed_dim, (1, 1))

        with torch.no_grad():
            self.k_proj_conv.weight.copy_(self.k_proj.weight.unsqueeze(-1).unsqueeze(-1))
            self.k_proj_conv.bias.copy_(self.k_proj.bias)
            self.v_proj_conv.weight.copy_(self.v_proj.weight.unsqueeze(-1).unsqueeze(-1))
            self.v_proj_conv.bias.copy_(self.v_proj.bias)
            self.q_proj_conv.weight.copy_(self.q_proj.weight.unsqueeze(-1).unsqueeze(-1))
            self.q_proj_conv.bias.copy_(self.q_proj.bias)
            self.out_proj_conv.weight.copy_(self.out_proj.weight.unsqueeze(-1).unsqueeze(-1))
            self.out_proj_conv.bias.copy_(self.out_proj.bias)
        self.k_proj_conv.to(self.k_proj.weight)
        self.v_proj_conv.to(self.v_proj.weight)
        self.q_proj_conv.to(self.q_proj.weight)
        self.out_proj_conv.to(self.out_proj.weight)
        self.mha_conv = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        if self.is_sha:
            return self.forward_sha(hidden_states)
        elif self.mha_conv:
            return self.forward_mha_conv(hidden_states = hidden_states,
                                attention_mask= attention_mask,
                                causal_attention_mask= causal_attention_mask,
                                output_attentions= output_attentions)
        return self.forward_mha_linear(hidden_states = hidden_states,
                                attention_mask= attention_mask,
                                causal_attention_mask= causal_attention_mask,
                                output_attentions= output_attentions)


    def forward_sha(self, x: torch.Tensor) -> torch.Tensor:
        """"input shape = (B, seq_len, emb_dim) = (1, 257, 1024) for 224x224 image inputs"""
        B, seq_len, emb_dim = x.shape
        x = x.permute(0, 2, 1).unsqueeze(-1) # (1, 257, 1024) -> (1, 1024, 257, 1)

        outs = []
        for ndx in range(self.num_heads):
            # (1, 1024, 257, 1) -> conv -> (1, 64, 257, 1) -> permute -> (1, 257, 1, 64)-> reshape -> (1, 257, 64)
            _query = self.q_convs[ndx](x).permute(0, 2, 3, 1).reshape(B, -1, self.head_dim)
            _key = self.k_convs[ndx](x).permute(0, 2, 3, 1).reshape(B, -1, self.head_dim)
            _value = self.v_convs[ndx](x).permute(0, 2, 3, 1).reshape(B, -1, self.head_dim)

            _attn = self.matmul_1[ndx]((_query * self.scale), _key.transpose(-2, -1))
            _attn = self.softmax(_attn)
            _attn = self.dropout(_attn)
            _out = self.matmul_2[ndx](_attn, _value) # -> (1, 257, 64)
            outs.append(_out)

        out = self.concat_1(*outs) # -> (1, 257, 1024)
        out = self.out_proj(out)
        return out, None


    def forward_mha_conv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()
        assert bsz == 1
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(-1) # input shape from 3d to 4d: (1, 77, 768) -> (1, 768, 77) -> (1, 768, 77, 1)

        # get query proj
        # output shape from 4d to 3d: (1, 768, 77, 1) -> (1, 768, 77) -> (1, 77, 768)
        query_states = self.q_proj_conv(hidden_states).squeeze(-1).permute(0, 2, 1) * self.scale
        key_states = self._shape(self.k_proj_conv(hidden_states).squeeze(-1).permute(0, 2, 1), -1, bsz)
        value_states = self._shape(self.v_proj_conv(hidden_states).squeeze(-1).permute(0, 2, 1), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        #attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = self.bmm_1(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        #attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.softmax(attn_weights)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        #attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = self.dropout(attn_weights)

        #attn_output = self.bmm_2(attn_probs, value_states)
        # Change to support 16-bit value (attn_probs [12, 14, 14], value_states [12, 14, 64], attn_output [12, 14, 64])
        attn_output = self.bmm_2(value_states.transpose(-1,-2), attn_probs.transpose(-1,-2)).transpose(-1,-2)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        # input shape from 3d to 4d: (1, 77, 768) -> (1, 768, 77) -> (1, 768, 77, 1)
        attn_output = attn_output.permute(0, 2, 1).unsqueeze(-1)
        attn_output = self.out_proj_conv(attn_output)
        # output shape from 4d to 3d: (1, 768, 77, 1) -> (1, 768, 77) -> (1, 77, 768)
        attn_output = attn_output.squeeze(-1).permute(0, 2, 1)
        return attn_output, attn_weights_reshaped


    def forward_mha_linear(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        #attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = self.bmm_1(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        #attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.softmax(attn_weights)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        #attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = self.dropout(attn_weights)

        #attn_output = self.bmm_2(attn_probs, value_states)
        # Change to support 16-bit value (attn_probs [12, 14, 14], value_states [12, 14, 64], attn_output [12, 14, 64])
        attn_output = self.bmm_2(value_states.transpose(-1,-2), attn_probs.transpose(-1,-2)).transpose(-1,-2)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


def replace_mha_with_sha_blocks(clip_model):
    print("linear layers in Attention will be replaced with convs with single head")
    for name, module in clip_model.named_modules():
        if isinstance(module, CLIPAttention):
            module._split_attention_heads()