#!/usr/bin/env python3
# ======================================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ======================================================================================

# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

""" This file provides adaptations to the Indus model. These adaptations are being done to optimize the model execution on the HTP backend. https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2 import modeling_gpt2
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def snr(signal, noisy, eps=1e-10):
    s = signal.pow(2).mean()
    n = (signal-noisy).pow(2).mean() + eps
    return 10 * torch.log10(s / n)

class QcGpt2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        GPT2Attention.__init__(self, config, is_cross_attention, layer_idx)
        self.use_unpack_qkv = False

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        #QC
        use_combined_mask_input = self.config.use_combined_mask_input if hasattr(self.config, 'use_combined_mask_input') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        #QC
        attn_weights = torch.matmul(query, key if transposed_key_cache else key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not use_combined_mask_input and not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        #QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        #QC

        if self.use_unpack_qkv:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query.shape[:-1], -1, self.head_dim)
        shape_kv = (*key.shape[:-1], -1, self.head_dim)

        query = query.view(shape_q).transpose(1, 2)
        key = key.view(shape_kv).transpose(1, 2)
        value = value.view(shape_kv).transpose(1, 2)

        #QC
        if transposed_key_cache:
            key = key.transpose(-1, -2)

        #QC
        if use_cache and return_new_key_value_only:
            present = (key, value)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-1 if transposed_key_cache else -2)
            value = torch.cat((past_value, value), dim=-2)

        #QC
        if not return_new_key_value_only:
            if use_cache is True:
                present = (key, value)
            else:
                present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        :param tensor: attn_output tensor and its shape is (bsz, num_attn_heads, seq_len, head_dim)

        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    # QC
    def unpack_qkv(self):
        self.use_unpack_qkv = True

        device = self.c_attn.weight.device
        self.q_proj = Conv1D(self.embed_dim, self.embed_dim).to(device)
        self.k_proj = Conv1D(self.embed_dim, self.embed_dim).to(device)
        self.v_proj = Conv1D(self.embed_dim, self.embed_dim).to(device)

        self.q_proj.weight.data.copy_(self.c_attn.weight[:, : self.embed_dim])
        self.k_proj.weight.data.copy_(self.c_attn.weight[:, self.embed_dim : 2 * self.embed_dim])
        self.v_proj.weight.data.copy_(self.c_attn.weight[:, 2 * self.embed_dim :])

        self.q_proj.bias.data.copy_(self.c_attn.bias[: self.embed_dim])
        self.k_proj.bias.data.copy_(self.c_attn.bias[self.embed_dim : 2 * self.embed_dim])
        self.v_proj.bias.data.copy_(self.c_attn.bias[2 * self.embed_dim :])

        #del self.c_attn


def GPT2Model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

    # QC
    use_combined_mask_input  = self.config.use_combined_mask_input if hasattr(self.config, 'use_combined_mask_input') else False

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    # Attention mask.
    _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
    attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None and not use_combined_mask_input else attention_mask
    if self._attn_implementation == "flash_attention_2":
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif _use_sdpa:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(batch_size, input_shape[-1]),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_length,
        )
    else:
        if attention_mask is not None and not use_combined_mask_input:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.add_cross_attention and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        if _use_sdpa:
            encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
            )
        elif not self._attn_implementation == "flash_attention_2":
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
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
        return_new_key_value_only = cache_kwargs.get('return_new_key_value_only', False)
        transposed_key_cache = cache_kwargs.get('transposed_key_cache', False)
        key_cat_dim = -1 if transposed_key_cache else -2

        key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=key_cat_dim)
        value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if return_new_key_value_only:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = key_cache
            self.value_cache[layer_idx] = value_cache
        return key_cache, value_cache


def DynamicCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.value_cache) <= layer_idx:
        return 0
    return self.value_cache[layer_idx].shape[-2]


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False