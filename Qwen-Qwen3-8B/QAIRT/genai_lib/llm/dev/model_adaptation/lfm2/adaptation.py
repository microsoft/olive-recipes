#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides adaptations to the LFM2 model. These adaptations are being done to
optimize the model execution on the HTP backend.
https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py"""

""" This file provides adaptations to the LFM2 model. These adaptations are being done to optimize the model execution on the HTP backend. """

import math
from typing import Any, Dict, Optional, Tuple, Union
from importlib.metadata import version
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import Lfm2ForCausalLM
from transformers.utils.generic import check_model_inputs
from transformers.masking_utils import create_causal_mask
from transformers.models.lfm2 import modeling_lfm2
from transformers.models.lfm2.modeling_lfm2 import (
    repeat_kv,
    Lfm2HybridConvCache,
    Lfm2Attention,
    Lfm2ShortConv,
    Lfm2Model,
    Lfm2Config,
    BaseModelOutputWithPast,
    apply_rotary_pos_emb,
)

from genai_lib.common.dev.utils import filter_outputs


from transformers.utils import logging
logger = logging.get_logger(__name__)


def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's lfm2, provided by Carl
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


class QcLfm2ShortConv(Lfm2ShortConv):
    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
    ):
        super().__init__(config, layer_idx)
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=0,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Lfm2HybridConvCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states * attention_mask
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, hidden_states = BCx.chunk(3, dim=-2)

        Bx = B * hidden_states

        if past_key_value is not None:
            conv_states = past_key_value.conv_cache[self.layer_idx].to(Bx.device)
            Bx = torch.concat([conv_states, Bx], dim=-1)
            if cache_position is not None and getattr(self.config, "shortconv_use_cache_position", False):
                indices = cache_position.expand(*Bx.shape[:-1], -1)
                conv_states = torch.gather(Bx, dim=-1, index=indices)
            else:
                if getattr(self.config, "shortconv_return_full_cache", False):
                    conv_states = Bx
                else:
                    conv_states = Bx[..., -self.L_cache + 1 :]
            conv_out = self.conv(Bx)
            past_key_value.conv_cache[self.layer_idx] = conv_states
        else:
            raise RuntimeError("should always have cache")

        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y

class QcLfm2ShortConvNative(Lfm2ShortConv):
    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
    ):
        super().__init__(config, layer_idx)
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=0,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Lfm2HybridConvCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = (hidden_states * attention_mask).transpose(-1, -2).unsqueeze(-2)
        BCx = self.in_proj(hidden_states)
        B, C, hidden_states = BCx.chunk(3, dim=-3)

        Bx = B * hidden_states

        if past_key_value is not None:
            conv_states = past_key_value.conv_cache[self.layer_idx]
            Bx = torch.concat([conv_states, Bx], dim=-1)
            if cache_position is not None and getattr(self.config, "shortconv_use_cache_position", False):
                indices = cache_position.expand(*Bx.shape[:-1], -1)
                conv_states = torch.gather(Bx, dim=-1, index=indices)
            else:
                if getattr(self.config, "shortconv_return_full_cache", False):
                    conv_states = Bx
                else:
                    conv_states = Bx[..., -self.L_cache + 1 :]
            conv_out = self.conv(Bx)
            past_key_value.conv_cache[self.layer_idx] = conv_states
        else:
            raise RuntimeError("should always have cache")

        y = C * conv_out
        y = self.out_proj(y).squeeze(-2).transpose(-1, -2)
        return y

    def post_init(self):
        print(f"[NativeConv Short Conv] update for layer {self.layer_idx}")
        dim = self.in_proj.in_features
        conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(1, self.L_cache),
            groups=dim,
            bias=False,
            padding=0,
        )
        conv.weight.data = self.conv.weight.data.unsqueeze(-2)
        self.conv = conv

        in_proj = nn.Conv2d(
            self.in_proj.in_features,
            self.in_proj.out_features,
            1,
            bias=self.bias
        )
        in_proj.weight.data = self.in_proj.weight.data[..., None, None]
        if self.bias:
            raise NotImplementedError()

        self.in_proj = in_proj

        out_proj = nn.Conv2d(
            self.out_proj.in_features,
            self.out_proj.out_features,
            1,
            bias=self.bias
        )
        out_proj.weight.data = self.out_proj.weight.data[..., None, None]
        if self.bias:
            raise NotImplementedError()

        self.out_proj = out_proj

class QcLfm2Attention(Lfm2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper
    """

    def __init__(self, config: Lfm2Config, layer_idx: int):
        super(QcLfm2Attention, self).__init__(config, layer_idx)

        # We only use "torch.where(attention_mask, input, min(input)-20)" sequence when the enable_masked_softmax is present in the config
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        key_states = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if cos.shape[-1] == query_states.shape[-1]:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states = _apply_rope_single(query_states, position_embeddings)
            key_states = _apply_rope_single(key_states, position_embeddings)


        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, Lfm2HybridConvCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
                        "num_key_value_heads": self.config.num_key_value_heads,
                        "head_dim": self.head_dim,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.shape[-1] != value_states.shape[-2]:
                attention_mask = attention_mask[:, :, :, : value_states.shape[-2]]
            if self.enable_masked_softmax:
                attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
                minus_value = -20
                attn_weights = torch.where(attention_mask==0, attn_weights, attn_weights_min + minus_value)
            else:
                attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, None

def Lfm2HybridConvCache_init(
        self,
        config: Lfm2Config,
        max_batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str, None] = None,
    ):
        self.key_cache = []
        self.value_cache = []
        self.max_batch_size = max_batch_size
        self.layer_types = config.layer_types
        self.first_attention_layer = self.layer_types.index("full_attention")
        self.conv_L_cache = config.conv_L_cache
        self._dtype = dtype

        self.conv_cache: list[torch.Tensor] = []
        device = torch.device(device) if device is not None else None

        for _ in range(config.num_hidden_layers):
            conv_state = torch.zeros(
                self.max_batch_size,
                config.hidden_size,
                # QC Adaptation: We only need to keep track of the past
                # conv_L_cache - 1 tokens instead of full since
                # we would use only at most conv_L_cache - 1 past tokens (decode mode)
                # by doing this, we could skip the slicing operation
                self.conv_L_cache-1,
                dtype=self._dtype,
                device=device,
            )
            # QC Adaptation
            if config.enable_shortconv_native:
                conv_state = conv_state.unsqueeze(2)
            torch._dynamo.mark_static_address(conv_state)
            self.conv_cache.append(conv_state)

def Lfm2HybridConvCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the cache
    if key_states is not None:
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(None)
                self.value_cache.append(None)
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


def Lfm2HybridConvCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    layer_idx = self.first_attention_layer if self.layer_types[layer_idx] != "full_attention" else layer_idx
    if len(self.value_cache) <= layer_idx or self.value_cache[layer_idx].numel() == 0:
        return 0
    return self.value_cache[layer_idx].shape[-2]

def from_legacy_cache(past_key_values, config, batch_size):
    """
    Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
    backward compatibility.
    """
    cache = Lfm2HybridConvCache(
        config=config, max_batch_size=batch_size
    )
    if past_key_values is not None:
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in config.attention_indices:
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
            else:
                conv_cache = past_key_values[layer_idx]
                cache.conv_cache[layer_idx] = conv_cache
    return cache


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False


class QcLfm2Model(Lfm2Model):
    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        conv_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        conv_cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            batch_size = inputs_embeds.shape[0]
            past_key_values = Lfm2HybridConvCache(
                config=self.config, max_batch_size=batch_size, dtype=self.dtype, device=self.device
            )

        if cache_position is None:
            past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length() if not isinstance(past_key_values, tuple) else past_key_values[self.config.attention_indices[0]][1].shape[-2]
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        if getattr(self.config, "shortconv_use_cache_position", False):
            assert conv_cache_position is not None, "Enabled shortconv use cache position, this should not be None!"

        hidden_states = inputs_embeds
        position_embeddings = self.pos_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            is_attention_layer = layer_idx in self.config.attention_indices
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask if is_attention_layer else conv_mask,
                past_key_value=past_key_values,
                cache_position=cache_position if is_attention_layer else conv_cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

orig_embedding_fwd = modeling_lfm2.Lfm2RotaryEmbedding.forward
def adapted_RotaryEmbedding(self, x, position_ids, *args, **kwargs):
    if isinstance(position_ids, tuple) and len(position_ids)==2:
        return position_ids
    else:
        return orig_embedding_fwd(self, x, position_ids, *args, **kwargs)


class QcLfm2ForCausalLM(Lfm2ForCausalLM):
    """
    Subclass of original Lfm2ForCausalLM, add to support cache_index input and cache management QC style.
    """
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference))


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            conv_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: Optional[int] = None,
            valid_token_mask: Optional[torch.Tensor]=None,
            cache_index: Optional[torch.Tensor]=None,
            conv_cache_position: Optional[torch.Tensor]=None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        num_logits_to_keep = num_logits_to_keep if num_logits_to_keep else getattr(self.config, "num_logits_to_keep", 0)

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcLfm2ForCausal doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor

        if isinstance(past_key_values, tuple):
            past_key_values = from_legacy_cache(past_key_values, self.config, input_ids.shape[0])

        if self.config.shortconv_use_cache_position:
            assert conv_cache_position is not None, f"{conv_cache_position=} should not be None if enable shortconv use cache position"

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Only for generation
        if conv_mask is None:
            conv_mask = attention_mask[..., None]
            if input_ids.shape[1] == 1:  # decode
                conv_mask = 1.

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            conv_cache_position=conv_cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if return_dict:
            outputs = CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
            )
        else:
            outputs = (logits, *outputs[1:])

        if return_dict:
            assert not isinstance(outputs[1], tuple)
            past_key_values_output = Lfm2HybridConvCache.to_legacy_cache(outputs.past_key_values, self.config.attention_indices, self.config.num_hidden_layers)
            outputs.past_key_values = past_key_values_output
        else:
            new_outputs = []
            for item in outputs:
                if isinstance(item, Lfm2HybridConvCache):
                    past_key_values_output = Lfm2HybridConvCache.to_legacy_cache(item, self.config.attention_indices, self.config.num_hidden_layers)
                    new_outputs.append(past_key_values_output)
                else:
                    new_outputs.append(item)
            outputs = tuple(new_outputs)

        if hasattr(self.config, "output_index_filter"):
            return filter_outputs(outputs, self.config.output_index_filter)
        return outputs

def Lfm2HybridConvCache_to_legacy_cache(self, attention_indices, num_hidden_layers):

    """
    Converts the Lfm2HybridConvCache instance to its equivalent in the legacy cache format for backward compatibility.

    The past_key_values passed into the model as input is a tuple.
    The Lfm2Model converts it into a Cache object if it isn't one already. Within the model, past_key_values flow as a Lfm2HybridConvCache object.
    Just before returning the output, the Lfm2Model converts the Lfm2HybridConvCache back to the legacy cache (tuple format).
    Since we added new attributes to our Cache object, we need to ensure they are included as additional entries in the returned tuple."""

    legacy_cache = ()
    for layer_idx in range(num_hidden_layers):
        if layer_idx in attention_indices:
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        else:
            legacy_cache += ((self.conv_cache[layer_idx]),)
    return legacy_cache
