#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""This file provides utilities that the pipeline needs to work with the adaptations made to the Mistral model."""

import torch
import functools
from importlib.metadata import version


def llm_update_causal_mask(prepared_1d_attn_mask, input_tensor, max_input_tokens, model_context_len, model_id_or_path, mask_neg=-100.0, cache_index=None, pad_to_left=True):
    '''
    Creates a causal mask (2D) from the 1D attention mask.

    params:
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor: input_ids / input_embeddings
    3. max_input_tokens: maximum number of tokens consumed per inference (equals ARN)
    4. model_context_len: maximum number of tokens the model can consume in total
    5. model_id_or_path: model name or path to pretrained model
    6. mask_neg: proxy for minus infinity (quantization-friendly large negative value)
    7. cache_index: starting position index for kv-cache
    8. pad_to_left: whether the KV cache is padded to the left or right
    '''
    mistral_model = _get_model(model_id_or_path)

    if pad_to_left:
        assert cache_index is None, (
            "Invalid argument: left padding + scatter KV update is not supported."
        )
    else:
        assert cache_index is not None, (
            "Invalid argument: right padding + concat KV update is not supported."
        )

    if cache_index is None:
        cache_position = torch.arange(model_context_len - max_input_tokens, model_context_len, device=input_tensor.device)
    else:
        cache_position = torch.arange(max_input_tokens, dtype=torch.float32, device=input_tensor.device) + cache_index.to(input_tensor.device)

    input_embeds = torch.ones((input_tensor.shape[0], input_tensor.shape[1], 1), device=input_tensor.device)
    # use_cache removed in transformers 4.48+, pass only supported kwargs
    _kwargs = dict(
        attention_mask=prepared_1d_attn_mask,
        input_tensor=input_embeds,
        cache_position=cache_position,
        past_key_values=None,
        output_attentions=True,
    )
    if version('transformers') < '4.48.0':
        _kwargs['use_cache'] = False
    prepared_attention_mask = mistral_model._update_causal_mask(**_kwargs)
    prepared_attention_mask = prepared_attention_mask.clamp_min(mask_neg)
    return prepared_attention_mask


def llm_create_position_embeddings(config, dtype=torch.float32, position_ids=None):
    '''
    Creates position embeddings (RoPE) from the position ids.

    params:
    1. config: model configuration used to create the MistralRotaryEmbedding object
    2. position_ids: position ids passed into the model
    '''
    hidden_size = config.hidden_size
    max_position_embeddings = config.max_position_embeddings
    num_attention_heads = config.num_attention_heads
    rope_theta = config.rope_theta
    dim = int(hidden_size // num_attention_heads)
    device = position_ids.device
    x = torch.ones(1, device=device, dtype=dtype)
    rotary_emb = _get_rotary_embedding(dim=dim, max_position_embeddings=max_position_embeddings, rope_theta=rope_theta, device=device, config=config)
    cos, sin = rotary_emb(x, position_ids=position_ids)
    cos, sin = cos.unsqueeze(dim=1), sin.unsqueeze(dim=1)
    cos = cos[:, :, :, :dim // 2]
    sin = sin[:, :, :, :dim // 2]
    return cos, sin


def _get_rotary_embedding(dim, max_position_embeddings, rope_theta, device, config=None):
    from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
    if version('transformers') >= '4.48.0':
        # Use explicit args — MistralRotaryEmbedding(config) broken in 4.50.1+
        rotary_emb = MistralRotaryEmbedding(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            device=device
        ).to(device)
    else:
        rotary_emb = MistralRotaryEmbedding(dim=dim, max_position_embeddings=max_position_embeddings, base=rope_theta, device=device)
    return rotary_emb


@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    from transformers.models.mistral.modeling_mistral import MistralModel
    config = AutoConfig.from_pretrained(model_id_or_path)
    config.num_hidden_layers = 1
    model = MistralModel(config)
    return model
