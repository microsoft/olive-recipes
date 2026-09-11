#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides utilities that the pipeline would need to work with the adaptations made to Mistral model. """

import torch
import functools
import importlib


def llm_update_causal_mask(prepared_1d_attn_mask, input_tensor, max_input_tokens, model_context_len, model_id_or_path, mask_neg = -100.0):
    '''

    This function creates a causal mask (2D) from the 1D attention mask

    params:
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor : input_ids/ input_embeddings
    3. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    4. model_context_len: maximum number of tokens that the model can consume in total
    5. model_id_or_path: #module name of baichuan under transformers_modules
    6. mask_neg: proxy for minus infinity since minus infinity is not quantization friendly. This value should be large
    enough to drown out tokens that should not be attended to
    '''
    baichuan_model = _get_model(model_id_or_path)
    input_embeds = torch.ones((input_tensor.shape[0], input_tensor.shape[1], 1), device = input_tensor.device)
    prepared_attention_mask = baichuan_model._prepare_decoder_attention_mask(attention_mask=prepared_1d_attn_mask, input_shape = (input_tensor.shape[0], input_tensor.shape[1]), inputs_embeds=input_embeds,past_key_values_length = model_context_len-max_input_tokens)
    prepared_attention_mask = prepared_attention_mask.clamp_min(mask_neg)
    return prepared_attention_mask

def llm_create_position_embeddings(config, model_id_or_path, position_ids=None):
    '''
    This function creates position embedding (RoPE) from the position ids.
    params:
    1. config: model configuration to create the LLamaRotaryEmbedding object, expect config for backward compatibility, in future transformers, we only expect to pass the config, the one we have in docker, takes in the req argument
    2. model_id_or_path: #module name of baichuan under transformers_modules
    3. position_ids: required position ids passed into the model
    '''
    max_position_embeddings = config.max_position_embeddings
    dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
    device = position_ids.device
    x = torch.ones(1, device=device)
    rotary_emb = _get_rotary_embedding(dim=dim, max_position_embeddings=max_position_embeddings, device=device, model_id_or_path=model_id_or_path)
    cos, sin = rotary_emb(x, seq_len=max_position_embeddings)
    # the cos and sin returned are of shape (1, 1, max_position_emb, dim), in order to index into the max_position_emb, we remove the first two dimensions.
    cos, sin = cos.squeeze(dim=1).squeeze(dim=0), sin.squeeze(dim=1).squeeze(dim=0)
    cos, sin = (torch.stack([cos[[position_ids[i]]] for i in range(position_ids.shape[0])]),
                torch.stack([sin[[position_ids[i]]] for i in range(position_ids.shape[0])]))
    cos, sin = cos.unsqueeze(dim = 1), sin.unsqueeze(dim = 1)
    cos = cos[:,:,:,:dim//2]
    sin = sin[:,:,:, :dim//2]
    return cos, sin

@functools.cache
def _get_rotary_embedding(dim, max_position_embeddings, device, model_id_or_path):
    # here, model_id_or_path represents the module name of Baichuan model under transformers_modules
    modeling_baichuan_name = model_id_or_path + '.modeling_baichuan'
    modeling_baichuan = importlib.import_module(modeling_baichuan_name)
    rotary_emb = modeling_baichuan.RotaryEmbedding(dim=dim, max_position_embeddings=max_position_embeddings, base=10000, device=device)
    return rotary_emb

@functools.cache
def _get_model(model_id_or_path):
    # here, model_id_or_path represents the module name of Baichuan model under transformers_modules
    modeling_baichuan_name = model_id_or_path + '.modeling_baichuan'
    modeling_baichuan = importlib.import_module(modeling_baichuan_name)
    config_path_name  = model_id_or_path + '.configuration_baichuan'
    config_path = importlib.import_module(config_path_name)
    config = config_path.BaichuanConfig()
    config.num_hidden_layers = 1
    model = modeling_baichuan.BaichuanModel(config)
    return model
