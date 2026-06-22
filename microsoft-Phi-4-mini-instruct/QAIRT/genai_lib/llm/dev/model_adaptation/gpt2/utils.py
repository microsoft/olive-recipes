#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch

def _update_causal_mask(attention_mask, input_tensor, cache_position, past_key_values):
    """
    The helper function which creates the 4d causal mask given a 2d attention mask.
    The below implementation is a simplified version of the _update_causal_mask found in huggingface transformers for Llama.
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor : input_ids/ input_embeddings
    3. cache_position : tensor that captures the positions of the valid input ids (query) in the given attention mask
    4. past_key_values : past kv sent into the model
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        return attention_mask

    dtype, device = input_tensor.dtype, input_tensor.device
    batch_size = input_tensor.shape[0]
    min_dtype = torch.finfo(dtype).min
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

    sequence_length = input_tensor.shape[1]
    target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )
    return causal_mask

def llm_update_causal_mask(prepared_1d_attn_mask, input_tensor, max_input_tokens, model_context_len, mask_neg = -100.0):
    '''

    This function creates a causal mask (2D) from the 1D attention mask

    params:
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor : input_ids/ input_embeddings
    3. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    4. model_context_len: maximum number of tokens that the model can consume in total
    5. mask_neg: proxy for minus infinity since minus infinity is not quantization friendly. This value should be large
    enough to drown out tokens that should not be attended to
    '''
    cache_position = torch.arange(model_context_len - max_input_tokens, model_context_len, device=input_tensor.device)
    input_embeds = torch.ones((input_tensor.shape[0], input_tensor.shape[1], 1), device=input_tensor.device)
    prepared_attention_mask = _update_causal_mask(attention_mask=prepared_1d_attn_mask,
                                                  input_tensor=input_embeds,
                                                  cache_position=cache_position,
                                                  past_key_values=None)
    prepared_attention_mask = prepared_attention_mask.clamp_min(mask_neg)
    return prepared_attention_mask