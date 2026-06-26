# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
#
# This file contains certain notices of software components included with the
# software that Qualcomm Technologies, Inc. ("QTI") is required to provide you.
# Except where prohibited by the open source license, the content of this file is
# provided solely to satisfy QTI's attribution and notice requirement; your use of
# these software components together with the QTI software ("Software") is subject
# to the terms of your license from QTI. Compliance with all copyright laws and
# software license agreements included in the notice section of this file are the
# responsibility of the user. Except as may be granted by separate express written
# agreement, this file provides no license to any patents, trademarks, copyrights,
# or other intellectual property of Qualcomm Incorporated or any of its
# subsidiaries.
#
# Software provided with this notice is NOT A CONTRIBUTION to any open source
# project. If alternative licensing is available for any of the components with
# licenses or attributions provided below, a license choice is made for receiving
# such code by QTI.

# Copyright (c) 2023 Qualcomm Technologies, Inc. All rights reserved.

# Qualcomm is a trademark of Qualcomm Incorporated, registered in the United
# States and other countries. All Qualcomm Incorporated trademarks are used with
# permission. Other products and brand names may be trademarks or registered
# trademarks of their respective owners.
#
# ==============================================================================
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
#
# =============================================================================

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.phi3.modeling_phi3 import (
    repeat_kv,
    Cache,
    DynamicCache,
    Phi3Attention,
    Phi3Config,
    apply_rotary_pos_emb,
    Phi3ForCausalLM,
    Phi3MLP,
)


def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    """
    Based on FacebookResearch's llama, provided by Carl
    """
    rope_real = rope_vals[
        0
    ]  # shape should be 1, 1, seqlen, head_dim * partial_rotary_factor
    rope_im = rope_vals[
        1
    ]  # shape should be 1, 1, seqlen, head_dim * partial_rotary_factor

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[..., : x.shape[-1] // 2]  # extract first half elements
    x_im = x[..., x.shape[-1] // 2 :]  # extract second half elements

    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real, x_prod_im), dim=3).view(*x.shape)
    return x


def bypass_Phi4RotaryEmbedding(self, x, position_ids, *args, **kwargs):
    return position_ids


class QcPhi4Attention(Phi3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # QC
        return_new_key_value_only = (
            self.config.return_new_key_value_only
            if hasattr(self.config, "return_new_key_value_only")
            else False
        )
        transposed_key_cache = (
            self.config.transposed_key_cache
            if hasattr(self.config, "transposed_key_cache")
            else False
        )
        partial_rotary_factor = (
            self.config.partial_rotary_factor
            if hasattr(self.config, "partial_rotary_factor")
            else 0.75
        )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.config.num_attention_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[
            ..., query_pos : query_pos + self.num_key_value_heads * self.head_dim
        ]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Partial rotary embedding
        rope_dim = int(partial_rotary_factor * self.head_dim)
        query_rot, query_pass = (
            query_states[..., :rope_dim],
            query_states[..., rope_dim:],
        )
        key_rot, key_pass = (
            key_states[..., :rope_dim],
            key_states[..., rope_dim:],
        )

        if isinstance(position_ids, (tuple, list)):  # QC
            rope_embedding = position_ids
            cos, sin = rope_embedding
            query_rot = _apply_rope_single(query_rot, rope_embedding)
            key_rot = _apply_rope_single(key_rot, rope_embedding)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_rot, key_rot = apply_rotary_pos_emb(
                query_rot, key_rot, cos, sin, position_ids
            )

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if transposed_key_cache:  # QC
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "return_new_key_value_only": return_new_key_value_only,
                "transposed_key_cache": transposed_key_cache,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:  # QC
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(
                self.head_dim
            )
        else:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(value_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.config.num_attention_heads,
            q_len,
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def prepare_conv(self):
        if not hasattr(self, "forward_no_conv"):
            self.q_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_attention_heads * self.head_dim,
                1,
                bias=False,
            )
            self.k_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                1,
                bias=False,
            )
            self.v_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                1,
                bias=False,
            )
            self.o_proj_conv = nn.Conv2d(
                self.config.num_attention_heads * self.head_dim,
                self.config.hidden_size,
                1,
                bias=False,
            )

            self.forward_no_conv = self.forward
            self.forward = self.forward_conv

            query_pos = self.config.num_attention_heads * self.head_dim
            kv_dim = self.num_key_value_heads * self.head_dim
            self.q_proj_conv.weight.data.copy_(
                self.qkv_proj.weight[:query_pos, :, None, None]
            )
            self.k_proj_conv.weight.data.copy_(
                self.qkv_proj.weight[query_pos : query_pos + kv_dim, :, None, None]
            )
            self.v_proj_conv.weight.data.copy_(
                self.qkv_proj.weight[query_pos + kv_dim :, :, None, None]
            )
            self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

            del self.qkv_proj

    def forward_conv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # QC
        return_new_key_value_only = (
            self.config.return_new_key_value_only
            if hasattr(self.config, "return_new_key_value_only")
            else False
        )
        transposed_key_cache = (
            self.config.transposed_key_cache
            if hasattr(self.config, "transposed_key_cache")
            else False
        )
        partial_rotary_factor = (
            self.config.partial_rotary_factor
            if hasattr(self.config, "partial_rotary_factor")
            else 0.75
        )

        bsz, q_len, _ = hidden_states.size()

        hidden_states = torch.reshape(
            hidden_states, (bsz, q_len, 1, self.config.hidden_size)
        ).transpose(1, 3)

        query_states = self.q_proj_conv(hidden_states)
        key_states = self.k_proj_conv(hidden_states)
        value_states = self.v_proj_conv(hidden_states)

        query_states = query_states.reshape(
            bsz, self.config.num_attention_heads, self.head_dim, q_len
        ).transpose(2, 3)
        key_states = key_states.reshape(
            bsz, self.num_key_value_heads, self.head_dim, q_len
        ).transpose(2, 3)
        value_states = value_states.reshape(
            bsz, self.num_key_value_heads, self.head_dim, q_len
        ).transpose(2, 3)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Partial rotary embedding
        rope_dim = int(partial_rotary_factor * self.head_dim)
        query_rot, query_pass = (
            query_states[..., :rope_dim],
            query_states[..., rope_dim:],
        )
        key_rot, key_pass = (
            key_states[..., :rope_dim],
            key_states[..., rope_dim:],
        )

        if isinstance(position_ids, (tuple, list)):  # QC
            rope_embedding = position_ids
            cos, sin = rope_embedding
            query_rot = _apply_rope_single(query_rot, rope_embedding)
            key_rot = _apply_rope_single(key_rot, rope_embedding)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_rot, key_rot = apply_rotary_pos_emb(
                query_rot, key_rot, cos, sin, position_ids
            )

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "return_new_key_value_only": return_new_key_value_only,
                "transposed_key_cache": transposed_key_cache,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:  # QC
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(
                self.head_dim
            )
        else:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.config.num_attention_heads,
            q_len,
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, 1, self.config.hidden_size)
        attn_output = attn_output.transpose(1, 3)
        attn_output = self.o_proj_conv(attn_output)
        attn_output = attn_output.transpose(1, 3)
        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def bypass_update_causal_mask(self, attention_mask, *args, **kwargs):
    # attention_mask is Causal mask and given as model input
    return attention_mask


def Phi4MLP_prepare_conv(self):
    if not hasattr(self, "forward_linear"):
        self.gate_proj_conv = nn.Conv2d(
            self.config.hidden_size, self.config.intermediate_size, 1, bias=False
        )
        self.up_proj_conv = nn.Conv2d(
            self.config.hidden_size, self.config.intermediate_size, 1, bias=False
        )
        self.down_proj_conv = nn.Conv2d(
            self.config.intermediate_size, self.config.hidden_size, 1, bias=False
        )
        self.forward_linear = self.forward
        self.forward = self.forward_conv

        self.gate_proj_conv.weight.data.copy_(
            self.gate_up_proj.weight[: self.config.intermediate_size, :, None, None]
        )
        self.up_proj_conv.weight.data.copy_(
            self.gate_up_proj.weight[self.config.intermediate_size :, :, None, None]
        )
        self.down_proj_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None])

        del self.gate_up_proj
        del self.down_proj


def Phi4MLP_forward_conv(self, x):
    bsz, _, _ = x.size()
    x = torch.reshape(x, (bsz, -1, 1, self.config.hidden_size))
    x = x.transpose(1, 3)  # Transpose right before and after Conv
    x = self.down_proj_conv(
        self.activation_fn(self.gate_proj_conv(x)) * self.up_proj_conv(x)
    )
    x = x.transpose(1, 3)
    x = torch.reshape(x, (bsz, -1, self.config.hidden_size))
    return x


def Phi4ForCausalLM_prepare_conv(self):
    if not hasattr(self, "lm_head_conv"):

        def lm_head_conv_forward(x):
            bsz, _, _ = x.size()
            x = torch.reshape(x, (bsz, -1, 1, self.config.hidden_size))
            x = x.transpose(1, 3)  # Transpose right before and after Conv
            x = self.lm_head_conv(x)
            x = x.transpose(1, 3)
            x = torch.reshape(x, (bsz, -1, self.config.vocab_size))
            return x

        self.lm_head_conv = nn.Conv2d(
            self.config.hidden_size, self.config.vocab_size, 1, bias=False
        )
        self.lm_head_conv.weight.data.copy_(self.lm_head.weight[:, :, None, None])

        del self.lm_head
        self.lm_head = lm_head_conv_forward


def Phi4ForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def DynamicCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += value_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    else:
        return_new_key_value_only = cache_kwargs.get("return_new_key_value_only", False)
        transposed_key_cache = cache_kwargs.get("transposed_key_cache", False)
        key_cat_dim = -1 if transposed_key_cache else -2

        key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=key_cat_dim)
        value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if return_new_key_value_only:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = key_cache
            self.value_cache[layer_idx] = key_cache
        return key_cache, value_cache


def DynamicCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.value_cache) <= layer_idx:
        return 0
    return self.value_cache[layer_idx].shape[-2]


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f"_original_{attr_name}"
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False
