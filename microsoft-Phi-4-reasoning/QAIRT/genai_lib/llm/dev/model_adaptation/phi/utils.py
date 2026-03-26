#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""  This file provides utilities that the pipeline would need to work with the adaptations made to Phi3 model. """

import torch
import functools

def llm_update_causal_mask(prepared_1d_attn_mask, input_tensor, max_input_tokens, model_context_len, model_id_or_path, mask_neg = -100.0, cache_index=None, pad_to_left = True):
    '''

    This function creates a causal mask (2D) from the 1D attention mask

    params:
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor : input_ids/ input_embeddings
    3. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    4. model_context_len: maximum number of tokens that the model can consume in total
    5. model_id: Model name or path to pretrained model
    6. mask_neg: proxy for minus infinity since minus infinity is not quantization friendly. This value should be large
    enough to drown out tokens that should not be attended to
    7. cache_index: the index for the starting position of kvcaches
    6. pad_to_left: determines if the KV cache is padded to the left or right
    '''
    phi_model = _get_model(model_id_or_path)

    # if the cache position is None, then we assume that the current input ids will be concatenated to the right end and
    # #hence we construct the cache position accordingly to be sent into the update_causal_mask

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

    input_embeds = torch.ones((input_tensor.shape[0], input_tensor.shape[1], 1), device = input_tensor.device)
    prepared_attention_mask = phi_model._update_causal_mask(attention_mask=prepared_1d_attn_mask, input_tensor=input_embeds, output_attentions=True,
                              cache_position=cache_position, past_key_values=None)
    prepared_attention_mask = prepared_attention_mask.clamp_min(mask_neg)
    return prepared_attention_mask

def llm_create_position_embeddings(config, position_ids=None):
    '''
    This function creates position embedding (RoPE) from the position ids.
    params:
    1. config: model configuration to create the Phi3RotaryEmbedding object, expect config for backward compatibility, in future transformers, we only expect to pass the config, the one we have in docker, takes in the req argument
    2. position_ids: required position ids passed into the model
    '''

    hidden_size = config.hidden_size
    max_position_embeddings = config.max_position_embeddings
    num_attention_heads = config.num_attention_heads
    rope_theta = config.rope_theta
    dim = int((hidden_size // num_attention_heads))
    device = position_ids.device
    x = torch.ones(1, device=device)
    rotary_emb = _get_rotary_embedding(dim=dim, max_position_embeddings=max_position_embeddings, rope_theta=rope_theta, device=device, config=config)

    cos, sin = rotary_emb(x, position_ids=position_ids)
    cos, sin = cos.unsqueeze(dim = 1), sin.unsqueeze(dim = 1)
    cos = cos[:,:,:,:dim//2]
    sin = sin[:,:,:, :dim//2]
    return cos, sin

def _get_rotary_embedding(dim, max_position_embeddings, rope_theta, device, config=None):
    from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding
    rotary_emb = Phi3RotaryEmbedding(config=config)
    return rotary_emb

@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    from transformers.models.phi3.modeling_phi3 import Phi3Model
    config = AutoConfig.from_pretrained(model_id_or_path)
    config.num_hidden_layers = 1
    model = Phi3Model(config)
    return model

