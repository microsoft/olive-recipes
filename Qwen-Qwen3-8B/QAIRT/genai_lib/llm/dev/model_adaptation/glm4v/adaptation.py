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
# Copyright 2024 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
#
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

""" This file provides adaptations to the GLM-4v model. These adaptations are being done to optimize the model execution on the HTP backend. https://huggingface.co/THUDM/glm-edge-v-2b, https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm/modeling_glm.py"""

import torch
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from genai_lib.llm.dev.model_adaptation.common.utils import QcUtilityMixin

# TODO: import the modeling_glm from transformers package when available
try:
    from transformers_modules.modeling_glm import (
        repeat_kv,
        Cache,
        DynamicCache,
        GlmAttention,
        GlmForCausalLM,
        GlmModel,
        GlmRotaryEmbedding,
        GlmConfig,
        apply_rotary_pos_emb
    )
except ImportError as e:
    print(f"{e} \n Please instantiate Glm4v with AutoModelForCausalLM first to have transformers_modules cache")

from genai_lib.common.dev.utils import filter_outputs

def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    """
        Perform rotary embedding based on the rope cos and sin values and return the embedded tensor
        Inputs:
            x: tensor, the tensor to be rotary-embedded
            rope_vals: tuple, a tuple with rope values (cos, sin)
    """

    rope_real = rope_vals[0]  # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1]  # shape should be 1, 1, seqlen, head_dim/2

    x_real = x[:, :, :, 0::2] # extract first half elements
    x_im = x[:, :, :, 1::2] # extract second half elements

    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    x = torch.cat((x_prod_real, x_prod_im), dim=3).view(*x.shape)
    return x


class QcGlmAttention(GlmAttention):
    def __init__(self, config: GlmConfig, layer_idx: Optional[int] = None):
        super(QcGlmAttention, self).__init__(config, layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config,
                                                                                     'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config,
                                                                           'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        # QC
        if cos.shape[-1] == query_states.shape[-1]:
            # cos and sin haven't split by half yet
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, partial_rotary_factor=self.partial_rotary_factor
            )
        else:
            query_states = _apply_rope_single(query_states, position_embeddings)
            key_states = _apply_rope_single(key_states, position_embeddings)

        # QC
        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        # QC
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                            "return_new_key_value_only": return_new_key_value_only,
                            "transposed_key_cache": transposed_key_cache}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # QC
        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states) * self.scaling
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # QC
        if attention_mask is not None:  # no matter the length, we just slice it
            if attention_mask.shape[-1] != value_states.shape[-2]:
                attention_mask = attention_mask[:, :, :, :value_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QcGlmForCausalLM(GlmForCausalLM, QcUtilityMixin):
    def __init__(self, config: GlmConfig):
        super().__init__(config)

    # QC: replace the forward pass order
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: Optional[torch.Tensor] = None,  # remove default value for MPP
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: Optional[int] = None,  # remove default value for MPP
            **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        num_logits_to_keep = num_logits_to_keep if num_logits_to_keep else getattr(self.config, "num_logits_to_keep", 0)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # QC
        if pixel_values is not None:
            batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            images=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        unfiltered = (logits,) + outputs[1:]

        idx_filter = getattr(self.config, "output_index_filter", None)
        if idx_filter is not None:
            filtered = filter_outputs(list(unfiltered), idx_filter)
        else:
            filtered = unfiltered

        if not return_dict:
            return ((loss,) + filtered) if loss is not None else filtered

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


orig_causal_mask = GlmModel._update_causal_mask
def adapted_update_causal_mask(self, attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    else:
        return orig_causal_mask(self, attention_mask, *args, **kwargs)


orig_embedding_fwd = GlmRotaryEmbedding.forward
def adapted_RotaryEmbedding(self, x, position_ids, *args, **kwargs):
    if isinstance(position_ids, tuple) and len(position_ids) == 2:
        return position_ids
    else:
        return orig_embedding_fwd(self, x, position_ids, *args, **kwargs)


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
