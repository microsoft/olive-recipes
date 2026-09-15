#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
This file provides utilities that the pipeline would need to work with the adaptations made to Qwen2_5_VL model.
Note that we support only transformers >= 4.54 in this version. To support earlier versions
    - Refer to our earlier model adaptations for the function llm_update_causal_mask and the adaptations file for the adapted_update_causal_mask method
"""

import torch
import functools
from transformers.masking_utils import create_causal_mask



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



# QWEN2_5_VL_ONBOARDING
# Utility function to access the `.get_rope_index` method of the `Qwen2_5_VLModel` class
def llm_get_rope_index(input_ids, model_id_or_path=None, model_config=None, image_grid_thw = None, video_grid_thw = None, second_per_grid_ts = None, attention_mask = None):
    if not ((model_id_or_path is None) ^ (model_config is None)):
        raise ValueError("Provide exactly one of `model_id_or_path` or `model_config`.")

    if model_config is not None:
        qwen2_5_vlmodel = _get_model_from_config(model_config)
    else:
        qwen2_5_vlmodel = _get_model(model_id_or_path)

    position_ids, rope_deltas = qwen2_5_vlmodel.get_rope_index(input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask)
    return position_ids, rope_deltas



def llm_create_position_embeddings(config, position_ids=None):
    '''
    This function creates stitched multimodal position embedding (MRoPE) from the position ids.
    params:
    1. config: model configuration to create the Qwen2_5_VLRotaryEmbedding object
    2. position_ids: required position ids passed into the model
    '''

    dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
    device = position_ids.device
    x = torch.ones(1, device=device)
    rotary_emb = _get_rotary_embedding(config = config, device = device)
    cos, sin = rotary_emb(x, position_ids=position_ids)
    mrope_section = config.rope_scaling["mrope_section"]

    # We stitch the obtained multimodal cos, sin together
    # This logic is similar to that apply_multimodal_rotary_pos_emb in modeling_qwen2_5_vl
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(dim=1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(dim=1)
    cos = cos[:,:,:,:dim//2]
    sin = sin[:,:,:,:dim//2]
    return cos, sin



def _get_rotary_embedding(config, device):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding
    rotary_emb = Qwen2_5_VLRotaryEmbedding(config = config, device = device)
    return rotary_emb



# QWEN2_5_VL_ONBOARDING
# New API to account for the extra dimension '3' in position_ids (3, batch_size, seq_len)
def llm_pad_mrope_position_ids(position_ids_slice, max_input_tokens, pad_value=0, pad_to_left = True):
    """
    This function pads the position_ids since slice may return position_ids that is smaller than what the model accepts (AR len)

    params:
    position_ids_slice: the current position_ids slice that is passed into the model in the current invocation
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    pad_value: padding value, this is defaulted to 0
    pad_to_left: boolean value indicating whether padding is done towards the left or right.

    """

    assert position_ids_slice is not None

    # Note that due to MRoPE, the position_ids have a shape of (3, batch_size, seq_len) in Qwen 2.5 VL
    assert position_ids_slice.dim() == 3

    _, batch_size, pos_ids_len = position_ids_slice.shape

    if pos_ids_len < max_input_tokens:
        # Note that the hardcoded value of 3 is hardcoded for Qwen 2.5 VL, and represents temporal, height, width
        pad_pos_ids = torch.full((3, batch_size, max_input_tokens-pos_ids_len), pad_value,
                                 dtype=position_ids_slice.dtype, device=position_ids_slice.device)

        if pad_to_left:
            position_ids = torch.cat((pad_pos_ids, position_ids_slice), dim=-1)
        else:
            position_ids = torch.cat((position_ids_slice, pad_pos_ids), dim=-1)

        return position_ids
    else:
        return position_ids_slice



# QWEN2_5_VL_ONBOARDING
# Preprocess text, images and videos and precompute inputs_embeds
def lmm_preprocess_inputs(
        input_ids = None,
        inputs_embeds = None,
        past_key_values = None,
        pixel_values = None,
        pixel_values_videos = None,
        image_grid_thw = None,
        video_grid_thw = None,
        embedding_layer = None,
        vision_model = None,
        image_token_id = None,
        video_token_id = None,
        spatial_merge_size = None,
        ):

    assert embedding_layer is not None, "`llm_preprocess_inputs` API needs the embedding layer/ its copy as an input"

    # We check if we are in pre-fill stage: if yes, we need to process images/videos if present
    kv_length = 0 if past_key_values is None else past_key_values.get_seq_length() if not isinstance(past_key_values, tuple) else past_key_values[0][1].shape[-2]
    if kv_length == 0:
        # We check that in the prefill stage, input_ids are provided and inputs_embeds is None since they are yet to be computed
        assert inputs_embeds is None, "inputs_embeds should be generated from the input_ids and image_embeds " \
                                      "(if provided) for the first inference (none kvcache mode)"
        assert input_ids is not None, f"input_ids should be provided for the first inference"

        # Obtain embeddings for the text inputs, we will have dummy embeddings in the place of images/videos which will be repopulated in the next part of the code
        inputs_embeds = embedding_layer(input_ids)

        # The remaining code follows the flow for computing image and video embeddings and scattering these to appropriate positions in inputs_embeds using the `input_ids`
        # The following code is taken from the modeling_qwen2_5_vl.Qwen2_5_VLModel forward function
        if pixel_values is not None:
            assert vision_model is not None, "`llm_preprocess_inputs` API needs the vision encoder/ its copy as an input if images/videos are present in the input"

            vision_model_dtype = next(vision_model.parameters()).dtype
            pixel_values = pixel_values.type(vision_model_dtype)
            # Obtain image embeddings, split embeddings for individual images
            image_embeds = vision_model(pixel_values, grid_thw=image_grid_thw)
            split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
            image_embeds = torch.split(image_embeds, split_sizes)
            image_embeds = torch.cat(image_embeds, dim=0)
            # Calculate the number of image tokens by counting the number of image token ids in input ids
            n_image_tokens = (input_ids == image_token_id).sum()
            n_image_features = image_embeds.shape[0]
            assert n_image_tokens == n_image_features, "Number of image tokens and features should match"

            # Create a mask to determine where to scatter the computed image embeddings
            mask = input_ids == image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            # Scatter the computed image embeddings to the locations dictated by the computed image mask
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            assert vision_model is not None, "`llm_preprocess_inputs` API needs the vision encoder/ its copy as an input if images/videos are present in the input"
            pixel_values_videos = pixel_values_videos.type(vision_model_dtype)
            # Obtain video embeddings, split embeddings for individual videos
            video_embeds = vision_model(pixel_values_videos, grid_thw=video_grid_thw)
            split_sizes = (video_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
            video_embeds = torch.split(video_embeds, split_sizes)
            video_embeds = torch.cat(video_embeds, dim=0)
            # Calculate the number of video tokens by counting the number of video token ids in input ids
            n_video_tokens = (input_ids == video_token_id).sum()
            n_video_features = video_embeds.shape[0]
            assert n_video_tokens != n_video_features, "Number of video tokens and features should match"

            # Create a mask to determine where to scatter the computed video embeddings
            mask = input_ids == video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            # Scatter the computed video embeddings to the locations dictated by the computed video mask
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds

    # In the decode stage, compute inputs_embeds if they have not been provided
    if inputs_embeds is None:
        inputs_embeds = embedding_layer(input_ids)
    return inputs_embeds



# QWEN2_5_VL_ONBOARDING
# Instantiate a model with one layer in the llm and vision encoder components, only used to access the `.get_rope_index` method
@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id_or_path)
    return _get_model_from_config(config)


def _get_model_from_config(model_config):
    """
    Create a minimal Qwen2_5_VLModel from a provided config.
    Makes a copy of the config and sets minimal layers for efficiency.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from copy import deepcopy
    minimal_config = deepcopy(model_config)

    minimal_config.num_hidden_layers = minimal_config.text_config.num_hidden_layers = minimal_config.vision_config.depth = 1

    model = Qwen2_5_VLModel(minimal_config)
    return model
