#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

# =============================================================================
# coding=utf-8
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
# =============================================================================

""" This file provides adaptations to the LLaMa model. These adaptations are being done to
optimize the model execution on the HTP backend.
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"""

""" This file provides adaptations to the LLaMa model. These adaptations are being done to optimize the model execution on the HTP backend. """

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from importlib.metadata import version
from importlib import util
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import LlamaForCausalLM
from transformers import cache_utils
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import (
    repeat_kv,
    Cache,
    DynamicCache,
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
)
from genai_lib.common.dev.utils import AIMET_TORCH_AVAILABLE

if AIMET_TORCH_AVAILABLE:
    from genai_lib.llm.long_context_utils import AnchorUpdaterKeySecond

from genai_lib.common.dev.utils import filter_outputs


from transformers.utils import logging
logger = logging.get_logger(__name__)


def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
    x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements

    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    return x


class QcLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper
    We override the init and initialize the anchor updater.
    The user can optionally pass in the alpha value to initialize the anchor_updater through the model config.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super(QcLlamaAttention, self).__init__(config, layer_idx)

        # We only initialize anchor_updater when the anchor_alpha is present in the config
        if getattr(config, "anchor_alpha", None) is not None:
            if not AIMET_TORCH_AVAILABLE:
                raise ValueError("Long Context is currently only supported in AIMET Torch")
            self.anchor_updater = AnchorUpdaterKeySecond(alpha=config.anchor_alpha)

        # We only use "torch.where(attention_mask, input, min(input)-20)" sequence when the enable_masked_softmax is present in the config
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #QC
        valid_token_mask = kwargs.get('valid_token_mask', None)
        anchor_buffer = kwargs.get('anchor_buffer', None)
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            if isinstance(position_ids, (tuple, list)): # QC
                position_embeddings  = position_ids
            else:
                position_embeddings = self.rotary_emb(value_states, position_ids)
        cos, sin = position_embeddings
        query_states = _apply_rope_single(query_states, position_embeddings)
        key_states = _apply_rope_single(key_states, position_embeddings)

        # we cache the un-transposed keys before we pass into the combined scoring model.
        untransposed_keys = key_states

        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
                        "num_key_value_heads": self.config.num_key_value_heads,
                        "head_dim": self.head_dim,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # key_states is the concatenated keys
        # past_key_value.key_cache[layer_idx] is the new key states
        # we invoke the scoring model and insert the outputs, anchor and the evict indices into the past_key_values (DynamicCache object) as additional attributes.
        if valid_token_mask is not None and anchor_buffer is not None:
            anchor_buffer = anchor_buffer[self.layer_idx]
            anchor = self.anchor_updater(new_keys = untransposed_keys, valid_token_mask=valid_token_mask, old_anchor = anchor_buffer)

            insert_meta_info_to_pastkv(past_key_values = past_key_value, meta_info_bundle={"anchor_buffer":anchor}, layer_idx=self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if self.enable_masked_softmax:
                attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
                minus_value = -20
                attn_weights = torch.where(attention_mask==0, attn_weights, attn_weights_min + minus_value)
            else:
                attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # handle version-specific return
        if version('transformers') >= '4.48.0':
            return attn_output, attn_weights
        else:
            return attn_output, attn_weights, past_key_value

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
        cache_position = cache_kwargs.get('cache_position')
        num_key_value_heads = cache_kwargs.get('num_key_value_heads')
        head_dim = cache_kwargs.get('head_dim')
        key_cat_dim = -1 if transposed_key_cache else -2

        # if the size of past key cache passed is smaller in value than the last position where the new kv is to be inserted
        # [in case when Cache position determined automatically by HF] (Ctx_len+ARN), then we want to perform concat and not do scattering.
        if self.value_cache[layer_idx].shape[-2] <= cache_position[-1]:
            key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=key_cat_dim)
            value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        else:
            # the cache_position passed in as model i/p by user is a 1d tensor reflecting the positions
            # from valid_kv_end to valid_kv_end+ARN, we convert this into the indices for scattering. [# bsz, num_key_value_heads, head_dim, seq_len]-> works for transposed keys
            indices = cache_position.view(1, 1, 1, -1).expand(value_states.shape[0], num_key_value_heads, head_dim, cache_position.shape[-1])

            value_cache = self.value_cache[layer_idx].scatter(dim=-2, index=indices.transpose(-1,-2), src=value_states)

            indices = indices.transpose(-1, -2) if key_cat_dim== -2 else indices
            key_cache = self.key_cache[layer_idx].scatter(dim=key_cat_dim, index=indices, src=key_states)


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


orig_causal_mask = modeling_llama.LlamaModel._update_causal_mask
def adapted_update_causal_mask(self, attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    else:
        return orig_causal_mask(self, attention_mask, *args, **kwargs)

orig_embedding_fwd = modeling_llama.LlamaRotaryEmbedding.forward
def adapted_RotaryEmbedding(self, x, position_ids, *args, **kwargs):
    if isinstance(position_ids, tuple) and len(position_ids)==2:
        return position_ids
    else:
        return orig_embedding_fwd(self, x, position_ids, *args, **kwargs)

class QcLlamaForCausalLM(LlamaForCausalLM):
    """
    Subclass of original LlamaForCausalLM. This is needed to serve two purposes:

    1. Starting from transformers version 4.45.0, the num_logits_to_keep argument is now required argument.
    Consequently, the prepared static graph will always include this additional argument.
    To maintain compatibility with our existing pipelines, we create a new class that inherits from
    LlamaForCausalLM. In this new class, we redefine the forward method without the num_logits_to_keep
    argument and in inside the forward we infer the num_logits_to_keep from the config and then call the superclass's forward method.

    2. For the Long Context scoring model within the LLM, we need to pass two additional arguments:
     anchor_buffer and valid_token_mask. These can be provided as keyword arguments (introduced in transformers version 4.47.0).
     However, this approach is incompatible with Onnx export, as Onnx does not support keyword arguments when creating the onnx graph.
     Therefore, we pass valid_token_mask and anchor_buffer as model inputs to LlamaForCausalLM,
     which in turn get recognized through keyword arguments in the downstream blocks.
      This is similar to how the DynamicCache object is not traced by jit.trace at the topmost level in LlamaForCausalLM.
      """
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference))


    def forward(
            self,
            input_ids: torch.LongTensor = None,
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
            num_logits_to_keep: Optional[int] = None,
            valid_token_mask: Optional[torch.Tensor]=None,
            anchor_buffer: Optional[torch.Tensor]=None,
            cache_index: Optional[torch.Tensor]=None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        num_logits_to_keep = num_logits_to_keep if num_logits_to_keep else getattr(self.config, "num_logits_to_keep", 0)

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcLlamaForCausal doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor

        if type(past_key_values) == tuple and version('transformers') >= '4.48.0':
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        outputs = super().forward(
            input_ids = input_ids,
            attention_mask= attention_mask,
            position_ids= position_ids,
            past_key_values= past_key_values,
            inputs_embeds= inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            valid_token_mask=valid_token_mask,
            anchor_buffer=anchor_buffer,
            **kwargs)

        if version('transformers') >= '4.48.0':
            if return_dict:
                assert type(outputs.past_key_values) != tuple
                past_key_values_output = DynamicCache.to_legacy_cache(outputs.past_key_values)
                outputs.replace(outputs, past_key_values = past_key_values_output)
            else:
                new_outputs = []
                for item in outputs:
                    if isinstance(item, DynamicCache):
                        past_key_values_output = DynamicCache.to_legacy_cache(item)
                        new_outputs.append(past_key_values_output)
                    else:
                        new_outputs.append(item)
                outputs = tuple(new_outputs)

        if hasattr(self.config, "output_index_filter"):
            return filter_outputs(outputs, self.config.output_index_filter)
        return outputs

def insert_meta_info_to_pastkv(
    past_key_values,
    meta_info_bundle: {},
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function adds two new model outputs to the DynamicCache object, eliminating the need to write a new adaptation for the additional output.

    params:
    past_key_values: The DynamicCache object
    evict_info_bundle: A dictionary containing the additional information to attach to the DynamicCache object.
    layer_idx: An attribute pointing to a list, where layer_idx corresponds to the data of the given layer.
    cache_kwargs: Additional keyword arguments for the cache.

    """
    for key, value in meta_info_bundle.items():
        if getattr(past_key_values, key, None) is None:
            setattr(past_key_values, key, [])

        key_attr = getattr(past_key_values, key, None)
        # Update the cache
        if len(key_attr) <= layer_idx:
            key_attr.append(value)
        else:
            key_attr[layer_idx] = value

def DynamicCache_to_legacy_cache(self):

    """
    Converts the DynamicCache instance to its equivalent in the legacy cache format for backward compatibility.

    The past_key_values passed into the model as input is a tuple.
    The LlamaModel converts it into a Cache object if it isn't one already. Within the model, past_key_values flow as a DynamicCache object.
    Just before returning the output, the LlamaModel converts the DynamicCache back to the legacy cache (tuple format).
    Since we added new attributes to our Cache object, we need to ensure they are included as additional entries in the returned tuple."""

    legacy_cache = ()
    for layer_idx in range(len(self)):
        legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
    if "anchor_buffer" in dir(self):
        return (legacy_cache, self.anchor_buffer)
    return legacy_cache
