#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
This file provides adaptations that the pipeline would need to work with the adaptations made to Qwen2_5_VL model.
Note that we support only transformers >= 4.54 in this version. To support earlier versions
    - Return past_key_values in QcQwen2_5_VLAttention
"""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from transformers import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    repeat_kv,
    DynamicCache,
    Qwen2_5_VLAttention,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLTextConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from importlib.metadata import version

from transformers.utils import logging
logger = logging.get_logger(__name__)

from genai_lib.llm.dev.model_adaptation.qwen2_5_vl.utils import llm_create_position_embeddings
from genai_lib.common.dev.utils import filter_outputs



def _apply_rope_single(x, rope_vals: tuple[torch.Tensor, torch.Tensor]):
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



# QWEN2_5_VL_ONBOARDING
class QcQwen2_5_VLAttention(Qwen2_5_VLAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        # We only use "torch.where(attention_mask, input, min(input)-20)" sequence when the enable_masked_softmax is present in the config
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        # QC Adaptation: From transformers>=4.56, past_key_value is deprecated in favor of past_key_values
        past_key_value = past_key_value if past_key_value is not None else past_key_values

        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        # Note - Adaption: Here we apply RoPE over query and key states separately due to hardware requirements.
        query_states = _apply_rope_single(query_states, position_embeddings)
        key_states = _apply_rope_single(key_states, position_embeddings)

        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
                        "num_key_value_heads": self.config.num_key_value_heads,
                        "head_dim": self.head_dim
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            if attention_mask.shape[-1]!=value_states.shape[-2]:
            # The following section of code will only run when we want to evaluate the adapted model, otherwise the last dimension will have shape mismatch between the attn_weights (=ARN) and the attention_mask (=CL)
                attention_mask= attention_mask[:, :, :, : value_states.shape[-2]]
            if self.enable_masked_softmax:
                attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
                minus_value = -20
                attn_weights = torch.where(attention_mask==0, attn_weights, attn_weights_min + minus_value)
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        # handle version-specific return
        if version('transformers') >= '4.54.0':
            return attn_output, attn_weights
        else:
            return attn_output, attn_weights, past_key_value



# QWEN2_5_VL_ONBOARDING
class QcQwen2_5_VLTextModel(Qwen2_5_VLTextModel):

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:


        # This code is copy pasted from the forward function for Qwen2_5_VLTextModel
        # Only parts concerned with position_ids are modified to account for
        # pre-computed position embeddings passed into the forward function
        # The changes are marked as ####QCAdaptation####

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )


        ####QcAdaptation####
        # If cos, sin were passed in as position_ids, catch them as position_embeddings
        if isinstance(position_ids, (tuple, list)):
            position_embeddings = position_ids
        else:
            # Else, we compute the position_embeddings
            # the hard coded `3` is for temporal, height and width.
            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            elif position_ids.dim() == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            position_embeddings = llm_create_position_embeddings(config = self.config, position_ids = position_ids)

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
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

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



def DynamicCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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



# QWEN2_5_VL_ONBOARDING
class QcQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference), persistent=False)


    def forward(
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
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[int] = None,
        cache_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)
        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), f"{self.__class__.__name__} doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor

        if isinstance(past_key_values, tuple):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        outputs = super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            pixel_values = pixel_values,
            pixel_values_videos = pixel_values_videos,
            image_grid_thw = image_grid_thw,
            video_grid_thw = video_grid_thw,
            rope_deltas = rope_deltas,
            cache_position = cache_position,
            second_per_grid_ts = second_per_grid_ts,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if return_dict:
            assert isinstance(outputs.past_key_values, DynamicCache)
            past_key_values_output = DynamicCache.to_legacy_cache(outputs.past_key_values)
            outputs.past_key_values = past_key_values_output
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
