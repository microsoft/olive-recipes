#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
# =============================================================================
# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""
QC Adaptation for Qwen3-VL's LLM part.
Validated on transformers==4.57.1
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Tuple

from transformers import DynamicCache, Cache
from transformers.utils import is_torchdynamo_compiling
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    repeat_kv,
    Qwen3VLForConditionalGeneration,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLTextModel,
    Qwen3VLTextAttention,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextConfig,
    apply_rotary_pos_emb,
)



__all__ = [
    "QcQwen3VLForConditionalGeneration",
    "QcQwen3VLModel",
    "QcQwen3VLTextModel",
    "QcQwen3VLTextAttention",
    "DynamicLayer_update",
    "DynamicLayer_get_seq_length",
]



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



class QcQwen3VLTextAttention(Qwen3VLTextAttention):

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # QC Adaptation: We only use "torch.where(attention_mask, input, min(input)-20)" sequence when the enable_masked_softmax is present in the config
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # QC Adaptation: Additional config options for hardware requirements.
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        if cos.shape[-1] == query_states.shape[-1]:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            # QC Adaptation: Apply RoPE over query and key states separately due to hardware requirements.
            query_states = _apply_rope_single(query_states, position_embeddings)
            key_states = _apply_rope_single(key_states, position_embeddings)

        # QC Adaptation: Transpose key states for better memory access pattern.
        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                        # QC Adaptation: Pass addtional arguments to update the cache.
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
                        "num_key_value_heads": self.config.num_key_value_heads,
                        "head_dim": self.head_dim
            }
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # QC Adaptation: Matmul with transposed key cache.
        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            if attention_mask.shape[-1] != value_states.shape[-2]:
                attention_mask = attention_mask[:, :, :, : value_states.shape[-2]]

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
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights



class QcQwen3VLTextModel(Qwen3VLTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`tuple[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape of each layer's deepstack embedding is (visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        hidden_states = inputs_embeds

        # QC Adaptation: Capture precomputed position embeddings.
        if isinstance(position_ids, (tuple, list)):
            position_embeddings = position_ids
            text_position_ids = None
        else:
            # else, run the original logic.
            # the hard coded `3` is for temporal, height and width.
            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
                position_ids = position_ids[1:]
            else:
                text_position_ids = position_ids[0]

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add deepstack visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                if visual_pos_masks is None:
                    # QC Adaptation: If we have preprocessed deepstack embeds using llm_masked_scatter_deepstack_embeds,
                    # we have already ensured size compatibility and hence can directly perform an add operation
                    hidden_states += deepstack_visual_embeds[layer_idx]
                else:
                    # This is the HF route of adding deepstack embeds
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )



class QcQwen3VLModel(Qwen3VLModel):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            deepstack_visual_embeds: Optional[tuple[torch.Tensor]] = None,
            **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        # QC Adaptation:
        # deepstack_visual_embeds = None  # We comment out this line from HF code as it force sets `deepstack_visual_embeds` to None - since we might be passing in precomputed deepstack embeds

        if deepstack_visual_embeds is None:
            if image_mask is not None and video_mask is not None:
                # aggregate visual_pos_masks and deepstack_visual_embeds
                image_mask = image_mask[..., 0]
                video_mask = video_mask[..., 0]
                visual_pos_masks = image_mask | video_mask
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
            elif image_mask is not None:
                image_mask = image_mask[..., 0]
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_mask is not None:
                video_mask = video_mask[..., 0]
                visual_pos_masks = video_mask
                deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )



class QcQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)

        # QC Adaptation: (Right padding) Register cache tensor.
        if getattr(config, "input_tokens_per_inference", None):
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference), persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # QC Adaptation: Updated type hint for legacy past_key_values.
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # QC Adaptation: Make logits_to_keep parameter as optional.
        logits_to_keep: Optional[Union[int, torch.Tensor]] = None,
        cache_index: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = False,
        deepstack_visual_embeds: Optional[tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:

        # QC Adaptation: Get logits_to_keep from config if not provided.
        logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)

        # QC Adaptation: (Right padding) Compute cache_position from cache_index and cache_tensor.
        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), f"{self.__class__.__name__} doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor

        # QC Adaptation: Convert legacy past_key_values to DynamicCache.
        if type(past_key_values) is tuple:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            return_dict=return_dict,
            deepstack_visual_embeds = deepstack_visual_embeds,
            **kwargs
        )

        # QC Adaptation: Convert DynamicCache back to legacy format for graph representation.
        if isinstance(outputs, tuple):
            outputs = tuple(
                item.to_legacy_cache() if isinstance(item, DynamicCache) else item
                for item in outputs
            )
        else:
            outputs.past_key_values = outputs.past_key_values.to_legacy_cache()

        return outputs



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



# QC Adaptation: Get length based on values instead of transposed keys. DynamicCache's get seq_len calls the layer's get_seq_length as --> self.layers[layer_idx].get_seq_length()
# Earlier we used to own the DynamicCache_update since keys and values were attribute of the Cache object, now we have these as part of the individual DynamicLayer object.
def DynamicLayer_get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.values.shape[-2]
