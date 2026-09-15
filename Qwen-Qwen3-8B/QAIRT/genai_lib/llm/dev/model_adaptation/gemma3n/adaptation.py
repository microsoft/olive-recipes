#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

# =============================================================================
# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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

""" This file provides adaptations to the Gemma3n model. These adaptations are being done to
optimize the model execution on the HTP backend.
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modeling_gemma3n.py"""

from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import Gemma3nForCausalLM, Gemma3nTextModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nTextConfig,
    Gemma3nTextAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from genai_lib.common.dev.utils import filter_outputs
logger = logging.get_logger(__name__)
def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):

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

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    transposed_key_cache=False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    # QC Adaptation: we define our own attention forward to work with both transposed and non-transposed Key cache.
    bsz, num_kv_heads, q_len, head_dim = query.shape
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if transposed_key_cache:
        attn_weights = torch.matmul(query, key_states) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        if attention_mask.shape[-1] != value_states.shape[-2]:
            attention_mask = attention_mask[:, :, :, : value_states.shape[-2]]

        # QC Adaptation: in order to enable masked_softmax, we need to set the enable_masked_softmax in the config which will add to the module when initializing attention block.
        if module.enable_masked_softmax:
            attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
            minus_value = -20
            attn_weights = torch.where(attention_mask==0, attn_weights, attn_weights_min + minus_value)
        else:
            attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, module.config.num_attention_heads, q_len, module.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, module.config.num_attention_heads, q_len, module.head_dim)}, but is"
                f" {attn_output.size()}"
            )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights



class QcGemma3nTextAttention(Gemma3nTextAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper
    """

    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super(QcGemma3nTextAttention, self).__init__(config, layer_idx)

        # We only use "torch.where(attention_mask, input, min(input)-20)" sequence when the enable_masked_softmax is present in the config
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #QC Adaptation: read the adaptation related items from config
        return_new_key_value_only = getattr(self.config, 'return_new_key_value_only', False)
        transposed_key_cache = getattr(self.config, 'transposed_key_cache', False)
        cos, sin = position_embeddings

        #QC Adaptation: For query and key states, in the HF flow, we first apply RoPE and then do the transpose, no change in the adaptations workflow as the _apply_rope_single is called after we do the transpose
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = self.q_norm(query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim))
        if cos.shape[-1] == query_states.shape[-1]:
            # expects non-transposed input- [bsz, seq_len, num_attention_heads, head_dim] and not [bsz, num_attention_heads, seq_len, head_dim]
            query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
            query_states = query_states.transpose(1, 2)
        else:
            query_states = query_states.transpose(1, 2)
            query_states = _apply_rope_single(query_states, position_embeddings)

        # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer
        if self.is_kv_shared_layer and past_key_values is not None:
            key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
            # Device of past layer may be different from current one
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            #QC
            """
            If we are computing the KV states for the non-shared layers,
            we need to do following
            1. proj on K,V
            2. norm -> transpose (in the adaptation flow, not in the HF flow for K, in HF the tranpose is after RoPE)
            3. rope on K
            """
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            key_states = self.k_norm(key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim))
            value_states = self.v_norm(value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)).transpose(1, 2)

            if cos.shape[-1] == key_states.shape[-1]:
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                key_states = key_states.transpose(1, 2)
            else:
                key_states = key_states.transpose(1, 2)
                key_states = _apply_rope_single(key_states, position_embeddings)

            # QC Adaptation: transpose key as needed
            if transposed_key_cache:
                key_states = key_states.transpose(2, 3)

        # QC Adaptation- need to ensure we are not running KV$ update for the shared layers, ops in the graph we do not want.
        if past_key_values is not None:
            assert isinstance(past_key_values, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
                        "num_key_value_heads": self.config.num_key_value_heads,
                        "head_dim": self.head_dim,
            }
            # Note: the following snippet is expected to work with >=4.56.0, in the previous versions HF performs an HF update even on the shared layers KV$ which is wasted op- https://github.com/huggingface/transformers/blob/e7d351cebad5f6dcdd169b0c034fdee0a000e6a9/src/transformers/models/gemma3n/modeling_gemma3n.py#L1306
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            transposed_key_cache=transposed_key_cache,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

class QcGemma3nTextModel(Gemma3nTextModel):

    def __init__(self, config: Gemma3nTextConfig):
        super(QcGemma3nTextModel, self).__init__(config)


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        swa_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        swa_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        swa_cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    )-> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (torch.Tensor, *optional*, defaults to None):
            Pre-computed per-layer embeddings. If None, they are derived from input_ids if provided.
        """
        # QC Adaptation: new inputs (swa_attention_mask,swa_position_ids,swa_cache_position) for handling the SWA/ local layers.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            """
            # QC Adaptation: we can have a case where we pass the per_layer_inputs and the input_ids.

                             per_layer_inputs_passed  | per_layer_inputs not passed
            inputs_embeds   | can support             | per layer inputs will be none inside the model, no model I/O
            input_ids       | can support             | per layer inputs will be computed inside the model from input ids, no model I/O
            """
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states_0 = inputs_embeds

        # QC Adaptation: Initialize RoPE embeddings from position_ids if not pre-computed outside the model.
        if isinstance(position_ids, (tuple, list)): # QC
            position_embeddings_global  = position_ids
            position_embeddings_local  = swa_position_ids
        else:
            position_embeddings_global = self.rotary_emb(hidden_states_0, position_ids)
            position_embeddings_local = self.rotary_emb_local(hidden_states_0, position_ids)


        # Expand hidden_states to support per-layer inputs
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5)

        temp_hidden_states = [hidden_states_0]
        for i in range(1, self.config.altup_num_inputs):
            # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_proj = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0)  # [num_altup_inputs, batch, seq_len, hidden_size]

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            per_layer_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]

            # QC Adaptation: We first check whether we are passing the swa_attention_mask, if we are, we are invoking the adapted model with adapted inputs. In that case we check whether the layer is sliding or not and then pass the relevant causal mask and the cache position.
            # If instead we are in the HF flow of adapted model, we pass the on-the-fly computed causal mask and the cache_position
            if swa_attention_mask is not None:
                if decoder_layer.self_attn.is_sliding:
                    cache_position_decoder = swa_cache_position
                    causal_mask = swa_attention_mask
                else:
                    cache_position_decoder = cache_position
                    causal_mask = attention_mask
            else:
                causal_mask = causal_mask_mapping[decoder_layer.attention_type]
                cache_position_decoder = cache_position

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global,
                position_embeddings_local,
                per_layer_input,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position_decoder,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer (but before reprojecting to stay consistent with layer output)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Per-layer inputs to single output
        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]
        for i in range(1, self.config.altup_num_inputs):
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj: torch.Tensor = self.altup_unembed_projections[i - 1](hidden_states[i])
            current_hidden_state = altup_unemb_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

# QC Adaptation: given our keys are transposed (we may need to perform right padding/ scatter), we need to own the DynamicLayer update function.
def DynamicLayer_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the cache as:  https://github.com/huggingface/transformers/blob/d79b2d981f28b2730d402244ac3c2e9a8c054eee/src/transformers/cache_utils.py#L98
    if self.keys is None:
        self.keys = key_states
        self.values = value_states
        return self.keys , self.values
    else:
        return_new_key_value_only = cache_kwargs.get('return_new_key_value_only', False)
        transposed_key_cache = cache_kwargs.get('transposed_key_cache', False)
        cache_position = cache_kwargs.get('cache_position')
        num_key_value_heads = cache_kwargs.get('num_key_value_heads')
        head_dim = cache_kwargs.get('head_dim')
        key_cat_dim = -1 if transposed_key_cache else -2
        # if the size of past key cache passed is smaller in value than the last position where the new kv is to be inserted
        # [in case when Cache position determined automatically by HF] (Ctx_len+ARN), then we want to perform concat and not do scattering.
        if self.values.shape[-2] <= cache_position[-1]:
            key_cache = torch.cat([self.keys, key_states], dim=key_cat_dim)
            value_cache = torch.cat([self.values, value_states], dim=-2)
        else:
            # the cache_position passed in as model i/p by user is a 1d tensor reflecting the positions
            # from valid_kv_end to valid_kv_end+ARN, we convert this into the indices for scattering. [# bsz, num_key_value_heads, head_dim, seq_len]-> works for transposed keys
            indices = cache_position.view(1, 1, 1, -1).expand(value_states.shape[0], num_key_value_heads, head_dim, cache_position.shape[-1])

            value_cache = self.values.scatter(dim=-2, index=indices.transpose(-1,-2), src=value_states)

            indices = indices.transpose(-1, -2) if key_cat_dim== -2 else indices
            key_cache = self.keys.scatter(dim=key_cat_dim, index=indices, src=key_states)

        if return_new_key_value_only:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = key_cache
            self.values = value_cache
        return key_cache, value_cache

# QC Adaptation: need to add this support given we will support from transformers >= 4.56, DynamicCache's get seq_len calls the layer's get_seq_length as --> self.layers[layer_idx].get_seq_length()
# Earlier we used to own the DynamicCache_update since keys and values were attribute of the Cache object, now we have these as part of the individual DynamicLayer object.
def DynamicLayer_get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.values.shape[-2]

class QcGemma3nForCausalLM(Gemma3nForCausalLM):
    """
    Subclass of original Gemma3nForCausalLM. This is needed to serve two purposes:

    1. Starting from transformers version 4.45.0, the num_logits_to_keep argument is now required argument.
    Consequently, the prepared static graph will always include this additional argument.
    To maintain compatibility with our existing pipelines, we create a new class that inherits from
    Gemma3nForCausalLM. In this new class, we redefine the forward method without the num_logits_to_keep
    argument and in inside the forward we infer the num_logits_to_keep from the config and then call the superclass's forward method.
      """
    def __init__(self, config):
        super().__init__(config)
        # QC Adaptation: to support the right padding/ scatter we need the cache_tensor
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference))


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            swa_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            swa_position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: Optional[int] = None,
            cache_index: Optional[torch.Tensor]=None,
            swa_cache_index: Optional[torch.Tensor]=None,
            per_layer_inputs: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # QC Adaptation: Note the new I/O
        num_logits_to_keep = num_logits_to_keep if num_logits_to_keep else getattr(self.config, "num_logits_to_keep", 0)
        return_dict = return_dict if return_dict else False
        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcGemma3nForCausalLM doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor

        # QC Adaptation: Following is initialized to ensure backwards HF compatibility
        swa_cache_position = None
        if swa_cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcGemma3nForCausalLM doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            swa_cache_position = swa_cache_index + self.cache_tensor
        if type(past_key_values) == tuple:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        outputs = super().forward(
            input_ids = input_ids,
            attention_mask= attention_mask,
            swa_attention_mask= swa_attention_mask,
            position_ids= position_ids,
            swa_position_ids= swa_position_ids,
            past_key_values= past_key_values,
            inputs_embeds= inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            swa_cache_position=swa_cache_position,
            num_logits_to_keep=num_logits_to_keep,
            per_layer_inputs=per_layer_inputs,
            **kwargs)

        if return_dict:
            assert type(outputs.past_key_values) != tuple
            outputs.past_key_values = outputs.past_key_values.to_legacy_cache()
        else:
            new_outputs = []
            for item in outputs:
                if isinstance(item, DynamicCache):
                    new_outputs.append(item.to_legacy_cache())
                else:
                    new_outputs.append(item)
            outputs = tuple(new_outputs)

        if hasattr(self.config, "output_index_filter"):
            return filter_outputs(outputs, self.config.output_index_filter)
        return outputs
