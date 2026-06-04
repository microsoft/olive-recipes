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

""" This file provides adaptations to the llama model. These adaptations are being 
done to optimize the model execution on the HTP backend. 
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"""

import types
import math
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import (
    repeat_kv,
    Cache,
    DynamicCache,
    LlamaAttention,
    LlamaMLP,
    LlamaConfig,
    apply_rotary_pos_emb,
)

from genai_lib.llm.dev.model_adaptation.llama.adaptation import _apply_rope_single
from aimet_torch.nn.modules import custom as aimet_ops

logger = logging.get_logger(__name__)

class LlamaMLPLora(LlamaMLP):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.lora_add_gate = aimet_ops.Add()
        self.lora_add_up = aimet_ops.Add()
        self.lora_add_down = aimet_ops.Add()
        self.lora_matmul_a_gate = aimet_ops.MatMul()
        self.lora_matmul_b_gate = aimet_ops.MatMul()
        self.lora_matmul_a_up = aimet_ops.MatMul()
        self.lora_matmul_b_up = aimet_ops.MatMul()
        self.lora_matmul_a_down = aimet_ops.MatMul()
        self.lora_matmul_b_down = aimet_ops.MatMul()
        self.lora_multiply_gate = aimet_ops.Multiply()
        self.lora_multiply_up = aimet_ops.Multiply()
        self.lora_multiply_down = aimet_ops.Multiply()
        self.lora_transpose_a_gate = aimet_ops.Permute()
        self.lora_transpose_b_gate = aimet_ops.Permute()
        self.lora_transpose_a_up = aimet_ops.Permute()
        self.lora_transpose_b_up = aimet_ops.Permute()
        self.lora_transpose_a_down = aimet_ops.Permute()
        self.lora_transpose_b_down = aimet_ops.Permute()

    def forward(self, x,
                lora_scale: float= None,
                lora_weights: Optional[Dict[str, torch.Tensor]] = None,
                layer_idx: int = None
                ):

        gate_lora_A = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.mlp.gate_proj.lora_A.weight") if lora_weights else None
        gate_lora_B = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.mlp.gate_proj.lora_B.weight") if lora_weights else None
        down_lora_A = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.mlp.down_proj.lora_A.weight") if lora_weights else None
        down_lora_B = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.mlp.down_proj.lora_B.weight") if lora_weights else None
        up_lora_A = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.mlp.up_proj.lora_A.weight") if lora_weights else None
        up_lora_B = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.mlp.up_proj.lora_B.weight") if lora_weights else None

        gate_proj_out = self.gate_proj(x)
        if gate_lora_A is not None:
            lora_gate_proj = self.lora_matmul_a_gate(x, self.lora_transpose_a_gate(gate_lora_A, [1, 0]))
            lora_gate_proj = self.lora_multiply_gate(lora_gate_proj, lora_scale)
            lora_gate_proj = self.lora_matmul_b_gate(lora_gate_proj, self.lora_transpose_b_gate(gate_lora_B, [1, 0]))
            gate_proj_out = self.lora_add_gate(gate_proj_out, lora_gate_proj)

        up_proj_out = self.up_proj(x)
        if up_lora_A is not None:
            lora_up_proj = self.lora_matmul_a_up(x, self.lora_transpose_a_up(up_lora_A, [1, 0]))
            lora_up_proj = self.lora_multiply_up(lora_up_proj, lora_scale)
            lora_up_proj = self.lora_matmul_b_up(lora_up_proj, self.lora_transpose_b_up(up_lora_B, [1, 0]))
            up_proj_out = self.lora_add_up(up_proj_out, lora_up_proj)

        act_fn_out = self.act_fn(gate_proj_out)
        pre_down_proj  = act_fn_out * up_proj_out
        down_proj_out = self.down_proj(pre_down_proj)
        if up_lora_A is not None:
            lora_down_proj = self.lora_matmul_a_down(pre_down_proj, self.lora_transpose_a_down(down_lora_A, [1, 0]))
            lora_down_proj = self.lora_multiply_down(lora_down_proj, lora_scale)
            lora_down_proj = self.lora_matmul_b_down(lora_down_proj, self.lora_transpose_b_down(down_lora_B, [1, 0]))
            down_proj_out = self.lora_add_down(down_proj_out, lora_down_proj)

        return down_proj_out

class LlamaAttentionLora(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.lora_add_q = aimet_ops.Add()
        self.lora_add_k = aimet_ops.Add()
        self.lora_add_v = aimet_ops.Add()
        self.lora_add_o = aimet_ops.Add()
        self.lora_matmul_a_q = aimet_ops.MatMul()
        self.lora_matmul_b_q = aimet_ops.MatMul()
        self.lora_matmul_a_k = aimet_ops.MatMul()
        self.lora_matmul_b_k = aimet_ops.MatMul()
        self.lora_matmul_a_v = aimet_ops.MatMul()
        self.lora_matmul_b_v = aimet_ops.MatMul()
        self.lora_matmul_a_o = aimet_ops.MatMul()
        self.lora_matmul_b_o = aimet_ops.MatMul()
        self.lora_multiply_q = aimet_ops.Multiply()
        self.lora_multiply_k = aimet_ops.Multiply()
        self.lora_multiply_v = aimet_ops.Multiply()
        self.lora_multiply_o = aimet_ops.Multiply()
        self.lora_transpose_a_q = aimet_ops.Permute()
        self.lora_transpose_b_q = aimet_ops.Permute()
        self.lora_transpose_a_k = aimet_ops.Permute()
        self.lora_transpose_b_k = aimet_ops.Permute()
        self.lora_transpose_a_v = aimet_ops.Permute()
        self.lora_transpose_b_v = aimet_ops.Permute()
        self.lora_transpose_a_o = aimet_ops.Permute()
        self.lora_transpose_b_o = aimet_ops.Permute()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        lora_scale: float= None,
        lora_weights: Optional[Dict[str, torch.Tensor]] = None,
        layer_idx: int = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        q_lora_A = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight") if lora_weights else None
        q_lora_B = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight") if lora_weights else None
        k_lora_A = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_A.weight") if lora_weights else None
        k_lora_B = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_B.weight") if lora_weights else None
        v_lora_A = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight") if lora_weights else None
        v_lora_B = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight") if lora_weights else None
        o_lora_A = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj.lora_A.weight") if lora_weights else None
        o_lora_B = lora_weights.get(
            f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj.lora_B.weight") if lora_weights else None

        if q_lora_A is not None:
            lora_q = self.lora_matmul_a_q(hidden_states, self.lora_transpose_a_q(q_lora_A, [1, 0]))
            lora_q = self.lora_multiply_q(lora_q, lora_scale)
            lora_q = self.lora_matmul_b_q(lora_q, self.lora_transpose_b_q(q_lora_B, [1, 0]))
            query_states = self.lora_add_q(query_states, lora_q)

        if k_lora_A is not None:
            lora_k = self.lora_matmul_a_k(hidden_states, self.lora_transpose_a_k(k_lora_A, [1, 0]))
            lora_k = self.lora_multiply_k(lora_k, lora_scale)
            lora_k = self.lora_matmul_b_k(lora_k, self.lora_transpose_b_k(k_lora_B, [1, 0]))
            key_states = self.lora_add_k(key_states, lora_k)

        if v_lora_A is not None:
            lora_v = self.lora_matmul_a_v(hidden_states, self.lora_transpose_a_v(v_lora_A, [1, 0]))
            lora_v = self.lora_multiply_v(lora_v, lora_scale)
            lora_v = self.lora_matmul_b_v(lora_v, self.lora_transpose_b_v(v_lora_B, [1, 0]))
            value_states = self.lora_add_v(value_states, lora_v)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            if isinstance(position_ids, (tuple, list)): # QC
                position_embeddings  = position_ids
            else:
                position_embeddings = self.rotary_emb(value_states, position_ids)
        cos, sin = position_embeddings
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
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if o_lora_A is not None:
            lora_o = self.lora_matmul_a_o(attn_output, self.lora_transpose_a_o(o_lora_A, [1, 0]))
            lora_o = self.lora_multiply_o(lora_o, lora_scale)
            lora_o = self.lora_matmul_b_o(lora_o, self.lora_transpose_b_o(o_lora_B, [1, 0]))
            attn_output = self.lora_add_o(attn_output, lora_o)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def forward_decoder(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    lora_scale: float= None,
    lora_weights: Optional[Dict[str, torch.Tensor]] = None,
    layer_idx: int = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        lora_scale=lora_scale,
        lora_weights=lora_weights,
        layer_idx=layer_idx,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states,
                             lora_scale=lora_scale,
                             lora_weights=lora_weights,
                             layer_idx=layer_idx)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def forward_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    lora_scale: Optional[float] = None,
    lora_weights: Optional[Dict[str, torch.Tensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (
        use_cache and not isinstance(past_key_values, Cache) and not self.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                lora_scale=lora_scale,
                lora_weights=lora_weights,
                layer_idx=layer_idx
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                lora_scale=lora_scale,
                lora_weights=lora_weights,
                layer_idx=layer_idx
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
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def forward_llama(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = None,
    lora_scale: Optional[float] = None,
    lora_weights: Optional[Dict[str, torch.Tensor]] = None,
    **kwargs
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        lora_scale=lora_scale,
        lora_weights=lora_weights
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

