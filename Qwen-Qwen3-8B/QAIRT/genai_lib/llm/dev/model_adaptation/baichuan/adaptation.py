#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

# ==============================================================================
# Copyright 2023 Baichuan Inc. All Rights Reserved.
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

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# ==============================================================================

""" This file provides adaptations to the Baichuan model. These adaptations are being done to optimize the model execution on the HTP backend. https://github.com/baichuan-inc/baichuan-7B/blob/main/models/modeling_baichuan.py"""

import math
from typing import List, Optional, Tuple, Union
from threading import Thread
import importlib

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerationConfig
from transformers.utils import logging, ContextManagers
import aimet_torch.elementwise_ops as op

module_name = 'transformers_modules.HF-Baichuan2-7B-Instruct.modeling_baichuan'
modeling_baichuan = importlib.import_module(module_name)
for attribute_name in dir(modeling_baichuan):
    if not attribute_name.startswith('__'):
        globals()[attribute_name] = getattr(modeling_baichuan, attribute_name)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    rope_real = rope_vals[0]  # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1]  # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:, :, :, :x.shape[-1] // 2]  # extract first half elements
    x_im = x[:, :, :, x.shape[-1] // 2:]  # extract second half elements

    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real, x_prod_im), dim=3)
    return x


class QCAttention(Attention):
    def __init__(self, config: PretrainedConfig):
        Attention.__init__(self, config)
        self.use_unpack_qkv = False

    def unpack_qkv(self):
        self.use_unpack_qkv = True

        device = self.W_pack.weight.device
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(device)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(device)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(device)

        # Adjust the slicing to match the shapes
        self.q_proj.weight.data.copy_(self.W_pack.weight[:self.hidden_size, :])
        self.k_proj.weight.data.copy_(self.W_pack.weight[self.hidden_size:2 * self.hidden_size, :])
        self.v_proj.weight.data.copy_(self.W_pack.weight[2 * self.hidden_size:, :])

    def scaled_dot_product_attention(self, query, key, value, attn_mask, transposed_key_cache):
        scale_factor = 1 / math.sqrt(query.size(-1))

        if transposed_key_cache:
            attn_weight = torch.matmul(query, key) * scale_factor
        else:
            attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

        attn_weight += attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, value)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # QC
        transposed_key_cache = (
            self.config.transposed_key_cache if hasattr(self.config, "transposed_key_cache") else False
        )
        return_new_key_value_only = (
            self.config.return_new_key_value_only if hasattr(self.config, "return_new_key_value_only") else False
        )

        bsz, q_len, _ = hidden_states.size()
        if self.use_unpack_qkv:
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # Added original fused QKV implementation just to be safe
            proj = self.W_pack(hidden_states)
            proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
            query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[1].shape[-2]

        if isinstance(position_ids, (tuple, list)):
            rope_embedding = position_ids
            query_states = apply_rope_single(query_states, rope_embedding)
            key_states = apply_rope_single(key_states, rope_embedding)
        else:
            cos, sin = self.rotary_emb(value_states, kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # QC
        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if return_new_key_value_only:
            present_key_value = (key_states, value_states) if use_cache else None

        if past_key_value is not None:
            dim = 3 if transposed_key_cache else 2
            key_states = torch.cat([past_key_value[0], key_states], dim=dim)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if not return_new_key_value_only:
            present_key_value = (key_states, value_states) if use_cache else None

        attn_output = self.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask,
                                                        transposed_key_cache)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


_prepare_decoder_attention_mask = modeling_baichuan.BaichuanModel._prepare_decoder_attention_mask
def adapted_prepare_decoder_attention_mask(self, attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    else:
        return _prepare_decoder_attention_mask(self, attention_mask, *args, **kwargs)


def QCBaichuanModel_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    elif isinstance(position_ids, (tuple, list)):
        pass
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False
