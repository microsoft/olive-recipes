#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

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

""" Eaglet speculative decoding algorithm https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py """
from typing import Optional, Tuple
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from genai_lib.llm.dev.model_adaptation.llama.adaptation import QcLlamaAttention
from genai_lib.common.dev.utils import is_transformers_greater_or_equal_than_4_48
from transformers.models.llama.configuration_llama import LlamaConfig
from .base_draft_model import BaseDraftModel, Eaglet2BaseDraftModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

__all__ = ["EagletDraftModel", "EagletDecoderLayer", "Eaglet2DraftModel"]


class EagletDraftModel(BaseDraftModel):
    def __init__(self, config):
        super().__init__(config=config, decoder_cls=EagletDecoderLayer)


class EagletDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = QcLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        if layer_idx > 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        if hasattr(self, "input_layernorm"):
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if is_transformers_greater_or_equal_than_4_48:
            hidden_states, self_attn_weights = self_attn_outputs
            # past_key_value is an instance of Cache having new KV cache after the attention
            present_key_value = past_key_value
        else:
            hidden_states, self_attn_weights, present_key_value = self_attn_outputs

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Eaglet2DraftModel(Eaglet2BaseDraftModel):
    def __init__(self, config):
        super().__init__(config=config, decoder_cls=EagletDecoderLayer)
