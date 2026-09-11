#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

# =============================================================================
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

""" Custom attention processors for improving on device latency and accuracy of Diffusers-based models https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py"""

from typing import Optional

import torch
from aimet_torch import elementwise_ops
from diffusers.models.attention_processor import Attention

from genai_lib.common.dev.model_adaptation.linear_to_conv import ConvInplaceLinear


class AttnProcessorConvSHA:
    """
    Convolution SHA processor for performing attention-related computations.
    Modified from source: https://github.com/huggingface/diffusers/blob/d7001400764acb8de5df343bbc4c54479c0e6ebe/src/diffusers/models/attention_processor.py#L710
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor, # 2, 4096, 640 / 2, 1024, 1280
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # 2, 77, 2048
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        head_dim = int(attn.inner_dim/attn.heads)
        args = () #if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        q_weight = attn.to_q.weight.reshape(attn.to_q.weight.shape[:2])
        k_weight = attn.to_k.weight.reshape(attn.to_k.weight.shape[:2])
        v_weight = attn.to_v.weight.reshape(attn.to_v.weight.shape[:2])
        proj_weight = attn.to_out[0].weight.reshape(attn.to_out[0].weight.shape[:2])
        scale_sqrt = attn.scale**0.5

        ### SHA ###
        hidden_states = hidden_states.unsqueeze(-1).permute(0, 2, 3, 1) # -> (B, C, 1, emb_dim)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(-1).permute(0, 2, 3, 1) # -> (B, C, 1, emb_dim)
        outs = []
        for idx in range(attn.heads):
            # (B, C, H, W) -> conv -> (B, head_dim, H, W)
            _query = torch.nn.functional.conv2d(hidden_states,
                                                weight=q_weight[idx*head_dim:(idx+1)*head_dim, :, None, None] * scale_sqrt,
                                                bias=attn.to_q.bias[idx*head_dim:(idx+1)*head_dim] * scale_sqrt if attn.to_q.bias is not None else None,
                                                ).permute(0, 2, 3, 1) # -> (B, H, W, head_dim)
            _key_T = torch.nn.functional.conv2d(encoder_hidden_states,
                                                weight=k_weight[idx*head_dim:(idx+1)*head_dim, :, None, None] * scale_sqrt,
                                                bias=attn.to_k.bias[idx*head_dim:(idx+1)*head_dim] * scale_sqrt if attn.to_k.bias is not None else None,
                                                ).reshape(batch_size, 1, head_dim, -1) # -> (B, 1, head_dim, H*W)
            _value = torch.nn.functional.conv2d(encoder_hidden_states,
                                                weight=v_weight[idx*head_dim:(idx+1)*head_dim, :, None, None],
                                                bias=attn.to_v.bias[idx*head_dim:(idx+1)*head_dim] if attn.to_v.bias is not None else None,
                                                ).permute(0, 2, 3, 1).reshape(batch_size, 1, -1, head_dim) # -> (B, 1, H*W, head_dim)

            _attn = elementwise_ops.MatMul()(_query, _key_T) # -> (B, H, W, H*W)
            _attn = torch.nn.functional.softmax(_attn, dim=-1) # -> (B, H, W, H*W)
            _out = elementwise_ops.MatMul()(_attn, _value) # -> (B, H, W, head_dim)
            outs.append(_out)

        hidden_states = elementwise_ops.Concat(-1)(*outs) # -> (B, H, W, C)
        ### END ###

        # linear proj
        hidden_states = hidden_states.permute(0, 3, 1, 2) # -> (B, C, H, W)
        hidden_states = torch.nn.functional.conv2d(hidden_states,
                                                   weight=proj_weight[:, :, None, None],
                                                   bias=attn.to_out[0].bias if attn.to_out[0].bias is not None else None,
                                                   ) # -> (B, C, H, W)
        hidden_states = hidden_states.permute(0, 2, 3, 1).squeeze(1) # -> (B, emb_dim, C)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttnProcessorWithoutMask:
    """
    Linear MHA processor without Attention Mask for performing attention-related computations.
    Modified from source: https://github.com/huggingface/diffusers/blob/d7001400764acb8de5df343bbc4c54479c0e6ebe/src/diffusers/models/attention_processor.py#L710
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () #if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        ### Without Mask  ###
        query = attn.to_q(hidden_states, *args)
        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_scores = torch.bmm(query * attn.scale, key.transpose(-1, -2))
        attention_probs = attention_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        ## END ###

        ### MHA reference ###
        # query = attn.to_q(hidden_states, *args)
        # key = attn.to_k(encoder_hidden_states, *args)
        # value = attn.to_v(encoder_hidden_states, *args)
        # query = attn.head_to_batch_dim(query)
        # key = attn.head_to_batch_dim(key)
        # value = attn.head_to_batch_dim(value)
        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = attn.batch_to_head_dim(hidden_states)
        ## END ###

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttnProcessorVAE:
    """
    Convolution tiled-SHA processor for slicing VAE-attention computations.
    Modified from source: https://github.com/huggingface/diffusers/blob/d7001400764acb8de5df343bbc4c54479c0e6ebe/src/diffusers/models/attention_processor.py#L710
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        inner_dim = 512,
        heads = 1,
    ) -> torch.Tensor:
        residual = hidden_states
        head_dim = int(inner_dim/heads)
        args = () #if USE_PEFT_BACKEND else (scale,)

        batch_size, channel, height, width = hidden_states.shape

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)#.transpose(1, 2))#.transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        q_weight = attn.to_q.weight.reshape(attn.to_q.weight.shape[:2])
        k_weight = attn.to_k.weight.reshape(attn.to_k.weight.shape[:2])
        v_weight = attn.to_v.weight.reshape(attn.to_v.weight.shape[:2])
        proj_weight = attn.to_out[0].weight.reshape(attn.to_out[0].weight.shape[:2])
        scale_sqrt = attn.scale**0.5

        ### SHA ###
        outs = []
        for idx in range(heads):
            # (B, C, H, W) -> conv -> (B, head_dim, H, W)
            _query = torch.nn.functional.conv2d(hidden_states,
                                                weight=q_weight[idx*head_dim:(idx+1)*head_dim, :, None, None] * scale_sqrt,
                                                bias=attn.to_q.bias[idx*head_dim:(idx+1)*head_dim] * scale_sqrt if attn.to_q.bias is not None else None,
                                                ).permute(0, 2, 3, 1) # -> (B, H, W, head_dim)
            _key_T = torch.nn.functional.conv2d(encoder_hidden_states,
                                                weight=k_weight[idx*head_dim:(idx+1)*head_dim, :, None, None] * scale_sqrt,
                                                bias=attn.to_k.bias[idx*head_dim:(idx+1)*head_dim] * scale_sqrt if attn.to_k.bias is not None else None,
                                                ).reshape(batch_size, 1, head_dim, -1) # -> (B, 1, head_dim, H*W)
            _value = torch.nn.functional.conv2d(encoder_hidden_states,
                                                weight=v_weight[idx*head_dim:(idx+1)*head_dim, :, None, None],
                                                bias=attn.to_v.bias[idx*head_dim:(idx+1)*head_dim] if attn.to_v.bias is not None else None,
                                                ).permute(0, 2, 3, 1).reshape(batch_size, 1, -1, head_dim) # -> (B, 1, H*W, head_dim)

            _attn = elementwise_ops.MatMul()(_query, _key_T) # -> (B, H, W, H*W)
            _attn = torch.nn.functional.softmax(_attn, dim=-1) # -> (B, H, W, H*W)
            _out = elementwise_ops.MatMul()(_attn, _value) # -> (B, H, W, head_dim)
            outs.append(_out)

        hidden_states = elementwise_ops.Concat(-1)(*outs) # -> (B, H, W, C)
        ### END ###

        # linear proj
        hidden_states = hidden_states.permute(0, 3, 1, 2) # -> (B, C, H, W)
        hidden_states = torch.nn.functional.conv2d(hidden_states,
                                                   weight=proj_weight[:, :, None, None],
                                                   bias=attn.to_out[0].bias if attn.to_out[0].bias is not None else None,
                                                   ) # -> (B, C, H, W)

        # dropout
        hidden_states = attn.to_out[1](hidden_states) # -> (B, C, H, W)

        if attn.residual_connection:
            hidden_states = hidden_states + residual # -> (B, C, H, W)
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttnProcessorUnet4dSHA:
    """
    Convolution SHA processor for performing attention-related computations.
    Modified from source: https://github.com/huggingface/diffusers/blob/d7001400764acb8de5df343bbc4c54479c0e6ebe/src/diffusers/models/attention_processor.py#L710
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor, # 2, 4096, 640 / 2, 1024, 1280
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # 2, 77, 2048
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        head_dim = int(attn.inner_dim/attn.heads)
        args = () #if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        batch_size, channel, height, width = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if encoder_hidden_states.ndim == 3:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(-1).permute(0, 2, 3, 1) # B, emb_dim, C -> B, C, 1, emb_dim

        q_weight = attn.to_q.weight.reshape(attn.to_q.weight.shape[:2])
        k_weight = attn.to_k.weight.reshape(attn.to_k.weight.shape[:2])
        v_weight = attn.to_v.weight.reshape(attn.to_v.weight.shape[:2])
        proj_weight = attn.to_out[0].weight.reshape(attn.to_out[0].weight.shape[:2])
        scale_sqrt = attn.scale**0.5

        ### SHA ###
        outs = []
        for idx in range(attn.heads):
            # (B, C, H, W) -> conv -> (B, head_dim, H, W)
            _query = torch.nn.functional.conv2d(hidden_states,
                                                weight=q_weight[idx*head_dim:(idx+1)*head_dim, :, None, None] * scale_sqrt,
                                                bias=attn.to_q.bias[idx*head_dim:(idx+1)*head_dim] * scale_sqrt if attn.to_q.bias is not None else None,
                                                ).permute(0, 2, 3, 1) # -> (B, H, W, head_dim)
            _key_T = torch.nn.functional.conv2d(encoder_hidden_states,
                                                weight=k_weight[idx*head_dim:(idx+1)*head_dim, :, None, None] * scale_sqrt,
                                                bias=attn.to_k.bias[idx*head_dim:(idx+1)*head_dim] * scale_sqrt if attn.to_k.bias is not None else None,
                                                ).reshape(batch_size, 1, head_dim, -1) # -> (B, 1, head_dim, H*W)
            _value = torch.nn.functional.conv2d(encoder_hidden_states,
                                                weight=v_weight[idx*head_dim:(idx+1)*head_dim, :, None, None],
                                                bias=attn.to_v.bias[idx*head_dim:(idx+1)*head_dim] if attn.to_v.bias is not None else None,
                                                ).permute(0, 2, 3, 1).reshape(batch_size, 1, -1, head_dim) # -> (B, 1, H*W, head_dim)

            _attn = elementwise_ops.MatMul()(_query, _key_T) # -> (B, H, W, H*W)
            _attn = torch.nn.functional.softmax(_attn, dim=-1) # -> (B, H, W, H*W)
            _out = elementwise_ops.MatMul()(_attn, _value) # -> (B, H, W, head_dim)
            outs.append(_out)

        hidden_states = elementwise_ops.Concat(-1)(*outs) # -> (B, H, W, C)
        ### END ###

        # linear proj
        hidden_states = hidden_states.permute(0, 3, 1, 2) # -> (B, C, H, W)
        hidden_states = torch.nn.functional.conv2d(hidden_states,
                                                   weight=proj_weight[:, :, None, None],
                                                   bias=attn.to_out[0].bias if attn.to_out[0].bias is not None else None,
                                                   ) # -> (B, C, H, W)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttnProcessorUnet4dMHA:
    """
    Convolution SHA processor for performing attention-related computations.
    Modified from source: https://github.com/huggingface/diffusers/blob/d7001400764acb8de5df343bbc4c54479c0e6ebe/src/diffusers/models/attention_processor.py#L710
    Forward with SHA but treat linears as convs
    """
    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor, # 2, 4096, 640 / 2, 1024, 1280
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # 2, 77, 2048
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        head_dim = int(attn.inner_dim/attn.heads)
        input_dim = hidden_states.ndim

        if attn.spatial_norm is not None: # None
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if attn.group_norm is not None: # None
            hidden_states = attn.group_norm(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size, channel, height, width = hidden_states.shape

        if encoder_hidden_states.ndim == 3:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(-1).permute(0, 2, 3, 1) # (B, L, D) -> (B, D, 1, L)/(B, C, 1, emb_dim)

        q_weight = attn.to_q.weight.reshape(attn.to_q.weight.shape[:2])
        k_weight = attn.to_k.weight.reshape(attn.to_k.weight.shape[:2])
        v_weight = attn.to_v.weight.reshape(attn.to_v.weight.shape[:2])
        proj_weight = attn.to_out[0].weight.reshape(attn.to_out[0].weight.shape[:2])

        ### MHA2SHA friendly MHA-Conv ###
        # Assuming hidden_state shape = (B, C, H, W)
        # Assuming encoder_hidden_states shape = (B, C, 1, emb_dim) or (B, C, H, W)
        # Returning shape (to match sha after concat) = (B, H, W, C)

        # All the ops in MHA will be replaced with sha in G2G, so inefficient ops will be cleaned up.
        # They are doing in such way that G2G can capture pattern better.
        # Requirement for a G2G friendly MHA-Conv:
        # 1) G2G uses the first reshape's shape[-2] after q_proj to determine head num.
        #    Remember to have the reshape after query in [..., head_num, head_dim]
        # 2) We have observed that ONNX will fold Mul in Matmul(Q, K) -> Mul -> Softmax to somewhere
        #    G2G do not have logic to search for scale in MatMul(Q*scale, K) pattern, so before the
        #    folding issue in Matmul(Q, K)*scale -> Softmax is clear, We should manually fold scale
        #    into query and key weights.
        #    e.g. Conv2d(x, q.weight) -> Conv2d(x, q.weight*scale), or
        #         Fold with sqrt: Conv2d(x, q.weight*sqrt(scale)) on both Q, and K for better numirical stability.
        #
        #
        # ONNX mha2sha converter will cleanup permute -> reshape after qkv_matmul.

        _query = torch.nn.functional.conv2d(hidden_states,
                                            weight=q_weight[..., None, None],
                                            bias=attn.to_q.bias,
                                            ).permute(0, 2, 3, 1).reshape(batch_size, -1, attn.heads, head_dim) # (B, C, H, W) -> (B, H, W, head_dim) -> (B, H*W, head, head_dim)
        _query = _query.permute(0, 2, 1, 3) # (B, H*W, head, head_dim) -> (B, head, H*W, head_dim)

        _key = torch.nn.functional.conv2d(encoder_hidden_states,
                                            weight=k_weight[..., None, None],
                                            bias=attn.to_k.bias,
                                            ).reshape(batch_size, attn.heads, head_dim, -1) # (B, D, 1, L) -> (B, head, d, L)
                                                                                            # (B, C, H, W) -> (B, head, d, H*W)

        _value = torch.nn.functional.conv2d(encoder_hidden_states,
                                            weight=v_weight[..., None, None],
                                            bias=attn.to_v.bias if attn.to_v.bias is not None else None,
                                            ).reshape(batch_size, attn.heads, head_dim, -1).permute(0, 1, 3, 2) # (B, D, 1, L) -> (B, head, d, L) -> (B, head, L, d)
                                                                                                                # # (B, C, H, W) -> (B, head, d, H*W) -> (B, head, H*W, d)
        _attn = elementwise_ops.MatMul()(_query * attn.scale, _key)  # (B, head, L, L)/(B, head, H*W, H*W)
        _attn = torch.nn.functional.softmax(_attn, dim=-1)
        _out = elementwise_ops.MatMul()(_attn, _value) # (B, head, L, d)/(B, head, H*W, head_dim)

        # These permute reshape will be cleaned up in MHA2SHA-onnx-converter
        _out = _out.permute(0, 2, 1, 3) # (B, head, H*W, d) -> (B, H*W, head, d)
        hidden_states = _out.reshape(batch_size, height, width, -1) # (B, H*W, head, d) -> (B, H, W, C)
        ### END ###

        # linear proj
        hidden_states = hidden_states.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        hidden_states = torch.nn.functional.conv2d(hidden_states,
                                                   weight=proj_weight[..., None, None],
                                                   bias=attn.to_out[0].bias if attn.to_out[0].bias is not None else None) # -> (B, C, H, W)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        hidden_states = hidden_states.reshape(batch_size, channel, height, width)
        return hidden_states


class AttnProcessorLoRA4D:
    """
    LoRA-Compatible processor for performing attention-related computations.
    Modified from source: https://github.com/huggingface/diffusers/blob/d7001400764acb8de5df343bbc4c54479c0e6ebe/src/diffusers/models/attention_processor.py#L710
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor, # 2, 4096, 640 / 2, 1024, 1280
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # 2, 77, 2048
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        head_dim = int(attn.inner_dim/attn.heads)

        if attn.spatial_norm is not None: # None
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if attn.group_norm is not None: # None
            hidden_states = attn.group_norm(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size, channel, height, width = hidden_states.shape

        if encoder_hidden_states.ndim == 3:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(-1).permute(0, 2, 3, 1) # (B, L, D) -> (B, D, 1, L)/(B, C, 1, emb_dim)

        if isinstance(attn.to_q, (torch.nn.Linear, ConvInplaceLinear)): # Linear expects channel's last
            _query = attn.to_q(hidden_states.permute(0, 2, 3, 1)).reshape(batch_size, -1, attn.heads, head_dim).permute(0, 2, 1, 3) # -> (B, head, H*W, head_dim)
        else: # Conv expects channel's first
            _query = attn.to_q(hidden_states).permute(0, 2, 3, 1).reshape(batch_size, -1, attn.heads, head_dim).permute(0, 2, 1, 3) # -> (B, head, H*W, head_dim)

        if isinstance(attn.to_k, (torch.nn.Linear, ConvInplaceLinear)): # Linear expects channel's last
            _key_T = attn.to_k(encoder_hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).reshape(batch_size, attn.heads, head_dim, -1) # -> (B, heads, head_dim, H*W)
        else: # Conv expects channel's first
            _key_T = attn.to_k(encoder_hidden_states).reshape(batch_size, attn.heads, head_dim, -1) # -> (B, heads, head_dim, H*W)

        if isinstance(attn.to_v, (torch.nn.Linear, ConvInplaceLinear)): # Linear expects channel's last
            _value = attn.to_v(encoder_hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).reshape(batch_size, attn.heads, head_dim, -1).permute(0, 1, 3, 2) # -> (B, heads, H*W, head_dim)
        else: # Conv expects channel's first
            _value = attn.to_v(encoder_hidden_states).reshape(batch_size, attn.heads, head_dim, -1).permute(0, 1, 3, 2) # -> (B, heads, H*W, head_dim)

        _attn = elementwise_ops.MatMul()(_query * attn.scale, _key_T) # -> (B, heads, H*W, H*W)
        _attn = torch.nn.functional.softmax(_attn, dim=-1) # -> (B, heads, H*W, H*W)
        _out = elementwise_ops.MatMul()(_attn, _value) # -> (B, heads, H*W, head_dim)

        # linear proj
        _out = _out.permute(0, 2, 1, 3) # (B, heads, H*W, head_dim) -> (B, H*W, heads, head_dim)
        hidden_states = _out.reshape(batch_size, height, width, -1) # (B, H*W, heads, head_dim) -> (B, H, W, C)
        hidden_states = hidden_states.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        if isinstance(attn.to_out[0], (torch.nn.Linear, ConvInplaceLinear)): # Linear expects channel's last
            hidden_states = attn.to_out[0](hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else: # Conv expects channel's first
            hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        hidden_states = hidden_states.reshape(batch_size, channel, height, width)
        return hidden_states

class AttnProcessorUnet4dMhaNamed:
    """
    Convolution SHA processor for performing attention-related computations.
    Modified from source: https://github.com/huggingface/diffusers/blob/d7001400764acb8de5df343bbc4c54479c0e6ebe/src/diffusers/models/attention_processor.py#L710
    Forward with SHA but treat linears as convs
    """
    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor, # 2, 4096, 640 / 2, 1024, 1280
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # 2, 77, 2048
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        head_dim = int(attn.inner_dim/attn.heads)
        input_dim = hidden_states.ndim

        if attn.spatial_norm is not None: # None
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if attn.group_norm is not None: # None
            hidden_states = attn.group_norm(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size, channel, height, width = hidden_states.shape

        if encoder_hidden_states.ndim == 3:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(-1).permute(0, 2, 3, 1) # (B, L, D) -> (B, D, 1, L)/(B, C, 1, emb_dim)

        _query = attn.to_q(hidden_states).permute(0, 2, 3, 1).reshape(batch_size, -1, attn.heads, head_dim).permute(0, 2, 1, 3) # (B, C, H, W) -> (B, H, W, head_dim) -> (B, H*W, head, head_dim) -> (B, head, H*W, head_dim)
        _key_T = attn.to_k(encoder_hidden_states).reshape(batch_size, attn.heads, head_dim, -1)  # (B, C, H, W) -> (B, head, d, H*W)
        _value = attn.to_v(encoder_hidden_states).reshape(batch_size, attn.heads, head_dim, -1).permute(0, 1, 3, 2) # (B, C, H, W) -> (B, head, d, H*W) -> (B, head, H*W, d)

        _attn = elementwise_ops.MatMul()(_query * attn.scale, _key_T)  # (B, head, L, L)/(B, head, H*W, H*W)
        _attn = torch.nn.functional.softmax(_attn, dim=-1)
        _out = elementwise_ops.MatMul()(_attn, _value) # (B, head, L, d)/(B, head, H*W, head_dim)

        # These permute reshape will be cleaned up in MHA2SHA-onnx-converter
        _out = _out.permute(0, 2, 1, 3) # (B, head, H*W, d) -> (B, H*W, head, d)
        hidden_states = _out.reshape(batch_size, height, width, -1) # (B, H*W, head, d) -> (B, H, W, C)
        ### END ###

        # linear proj
        hidden_states = hidden_states.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        hidden_states = attn.to_out[0](hidden_states) # -> (B, C, H, W)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        hidden_states = hidden_states.reshape(batch_size, channel, height, width)
        return hidden_states


def replace_attention_linear_with_conv(module):
    '''
    Replace linear layers in attention with conv2d layers. This function will also align the module's qkvo layer
    names with that in AttnProcessorLoRA4D for ease of loading base encodings. When this function is used to do
    linear to conv adaptation in attention layers, AttnProcessorUnet4dMhaNamed should be used as attention processor.

    :param module: The attention module in which linears are to be replaced
    :return: None. Replacement are in-place
    '''
    q_weight = module.to_q.weight.reshape(module.to_q.weight.shape[:2])
    q_bias = module.to_q.bias
    module.to_q = torch.nn.Conv2d(q_weight.shape[1], q_weight.shape[0], kernel_size=1,
                                  bias=module.to_q.bias is not None, device=q_weight.device)
    module.to_q.weight.data = q_weight[:, :, None, None]
    if module.to_q.bias is not None:
        module.to_q.bias.data = q_bias

    k_weight = module.to_k.weight.reshape(module.to_k.weight.shape[:2])
    k_bias = module.to_k.bias
    module.to_k = torch.nn.Conv2d(k_weight.shape[1], k_weight.shape[0], kernel_size=1,
                                  bias=module.to_k.bias is not None, device=k_weight.device)
    module.to_k.weight.data = k_weight[:, :, None, None]
    if module.to_k.bias is not None:
        module.to_k.bias.data = k_bias

    v_weight = module.to_v.weight.reshape(module.to_v.weight.shape[:2])
    v_bias = module.to_v.bias
    module.to_v = torch.nn.Conv2d(v_weight.shape[1], v_weight.shape[0], kernel_size=1,
                                  bias=module.to_v.bias is not None, device=v_weight.device)
    module.to_v.weight.data = v_weight[:, :, None, None]
    if module.to_v.bias is not None:
        module.to_v.bias.data = v_bias

    proj_weight = module.to_out[0].weight.reshape(module.to_out[0].weight.shape[:2])
    proj_bias = module.to_out[0].bias
    module.to_out[0] = torch.nn.Conv2d(proj_weight.shape[1], proj_weight.shape[0], kernel_size=1,
                                       bias=module.to_out[0].bias is not None, device=proj_weight.device)
    module.to_out[0].weight.data = proj_weight[:, :, None, None]
    if module.to_out[0].bias is not None:
        module.to_out[0].bias.data = proj_bias
