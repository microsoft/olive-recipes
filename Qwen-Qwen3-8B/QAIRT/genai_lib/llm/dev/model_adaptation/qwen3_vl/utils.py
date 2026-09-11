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
Utility functions for adapted Qwen3-VL's LLM part.
Validated on transformers==4.57.1
"""

import torch
import functools
from transformers.masking_utils import create_causal_mask


__all__ = [
    "llm_create_causal_mask",
    "llm_get_rope_index",
    "llm_pad_mrope_position_ids",
    "llm_create_position_embeddings",
    "llm_masked_scatter_deepstack_embeds",
]


def llm_create_causal_mask(prepared_1d_attn_mask, input_tensor, max_input_tokens, model_context_len, text_config, mask_neg = -1e3, cache_index = None, pad_to_left = True):
    '''

    This function creates a causal mask (2D) from the 1D attention mask

    params:
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor : input_ids/ input_embeddings
    3. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    4. model_context_len: maximum number of tokens that the model can consume in total
    5. text_config: Text config of the model
    6. mask_neg: proxy for minus infinity since minus infinity is not quantization friendly. This value should be large
    enough to drown out tokens that should not be attended to
    7. cache_index: the index for the starting position of kvcaches
    8. pad_to_left: determines if the KV cache is padded to the left or right
    '''

    # if the cache position is None, then we assume that the current input ids will be concatenated to the right end and
    # #hence we construct the cache position accordingly to be sent into the create_causal_mask

    if pad_to_left:
        # Cache index should not be passed. Concat op is used in doing the KV cache update
        assert cache_index is None, "Invalid argument error: we do not support the combination of performing left padding and doing scatter for KV cache update."
    else:
        # if the user is doing right padding, it is necessary to pass the cache_index.
        assert cache_index is not None, "Invalid argument error: we do not support the combination of performing right padding and doing concat for KV cache update"

    if cache_index is None:
        cache_position = torch.arange(model_context_len-max_input_tokens, model_context_len, device = input_tensor.device)
    else:
        cache_position = torch.arange(max_input_tokens, dtype=torch.float32, device=input_tensor.device) + cache_index.to(input_tensor.device)

    input_embeds = torch.ones((input_tensor.shape[0], model_context_len, 1), device = input_tensor.device)
    mask_kwargs = {
        "config": text_config,
        "input_embeds": input_embeds,
        "attention_mask": prepared_1d_attn_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": None,
    }
    prepared_attention_mask = create_causal_mask(**mask_kwargs)
    prepared_attention_mask = prepared_attention_mask.clamp_min(mask_neg)
    return prepared_attention_mask


def llm_get_rope_index(
    input_ids, model_id_or_path, image_grid_thw=None, video_grid_thw=None, attention_mask=None
):
    """
    This function gets the multimodal position ids and rope deltas from the vl model's get_rope_index method.
    """

    vl_model = _get_model(model_id_or_path)
    position_ids, rope_deltas = vl_model.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
    return position_ids, rope_deltas


def llm_pad_mrope_position_ids(position_ids_slice, max_input_tokens, pad_value=0, pad_to_left=True):
    """
    This function pads the position_ids since slice may return position_ids that is smaller than what the model accepts (AR len)
    Account for the MRoPE dimension '3'(temporal, height and width) in position_ids with shape of (3, batch_size, seq_len)

    params:
    position_ids_slice: the current position_ids slice that is passed into the model in the current invocation
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    pad_value: padding value, this is defaulted to 0
    pad_to_left: boolean value indicating whether padding is done towards the left or right.

    """

    assert position_ids_slice is not None

    # Note that due to MRoPE, the position_ids have a shape of (3, batch_size, seq_len) in Qwen 3 VL
    assert position_ids_slice.dim() == 3

    _, batch_size, pos_ids_len = position_ids_slice.shape

    if pos_ids_len < max_input_tokens:
        # Note that the hardcoded value of 3 is hardcoded for Qwen 3 VL, and represents temporal, height, width
        pad_pos_ids = torch.full((3, batch_size, max_input_tokens-pos_ids_len), pad_value,
                                 dtype=position_ids_slice.dtype, device=position_ids_slice.device)

        if pad_to_left:
            position_ids = torch.cat((pad_pos_ids, position_ids_slice), dim=-1)
        else:
            position_ids = torch.cat((position_ids_slice, pad_pos_ids), dim=-1)

        return position_ids
    else:
        return position_ids_slice


def llm_create_position_embeddings(text_config, position_ids):
    '''
    This function creates stitched multimodal position embedding (MRoPE) from the position ids.
    params:
    1. text_config: text model configuration to create the RotaryEmbedding object
    2. position_ids: required position ids passed into the model
    '''

    dim = text_config.head_dim if hasattr(text_config, 'head_dim') else text_config.hidden_size // text_config.num_attention_heads
    device = position_ids.device
    x = torch.ones(1, device=device)
    rotary_emb = _get_rotary_embedding(text_config=text_config, device=device)
    cos, sin = rotary_emb(x, position_ids=position_ids)

    cos, sin = cos.unsqueeze(dim=1), sin.unsqueeze(dim=1)
    cos = cos[:, :, :, : dim // 2]
    sin = sin[:, :, :, : dim // 2]
    return cos, sin


def llm_masked_scatter_deepstack_embeds(
    inputs_embeds,
    deepstack_visual_embeds,
    visual_pos_masks,
):
    """
    Scatter DeepStack visual embeddings into positions indicated by a visual mask,
    producing tensors that match the shape of `inputs_embeds`.

    Parameters
    ----------
    inputs_embeds
        Shape: (batch_size, seq_len, emb_dim)
        Reference tensor whose shape and device are used for output allocation.

    deepstack_visual_embeds
        Tuple of tensors, each with shape: (num_img_tokens, emb_dim).
        Visual deepstack embedding stacks to scatter into masked positions.

    visual_pos_masks
        Shape: (batch_size, seq_len).
        Boolean mask indicating which positions correspond to visual tokens.

    Returns
    -------
    Tuple of tensors, each with shape: (batch_size, seq_len, emb_dim),
    where visual positions contain the corresponding DeepStack embeddings and
    all non-visual positions are zero.
    """
    image_mask = visual_pos_masks.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    scattered_deepstack_embeds = [torch.zeros(inputs_embeds.shape, device=inputs_embeds.device) for _ in range(len(deepstack_visual_embeds))]
    scattered_deepstack_embeds = [t.masked_scatter(image_mask, embed) for t, embed in zip(scattered_deepstack_embeds, deepstack_visual_embeds)]
    return scattered_deepstack_embeds


@functools.cache
def _get_model(model_id_or_path):
    """
    Instantiate a model with one layer in the llm and vision encoder components, only used at [llm_get_rope_index] method
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    config = AutoConfig.from_pretrained(model_id_or_path)
    config.num_hidden_layers = config.text_config.num_hidden_layers = config.vision_config.depth = 1
    model = Qwen3VLModel(config)
    return model


def _get_rotary_embedding(text_config, device):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    rotary_emb = Qwen3VLTextRotaryEmbedding(config = text_config, device = device)
    return rotary_emb
