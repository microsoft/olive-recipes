#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides adaptations to the MLLaMa model. These adaptations are being done to optimize the model execution on the HTP backend. """

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.models.mllama import modeling_mllama
from transformers.models.mllama.modeling_mllama import (
    repeat_kv,
    Cache,
    DynamicCache,
    MllamaTextSelfAttention,
    MllamaTextCrossAttention,
    MllamaForConditionalGeneration,
    MllamaForCausalLM,
    MllamaConfig,
    MllamaTextConfig,
    _prepare_cross_attention_mask,
    MllamaCrossAttentionDecoderLayer
)

from genai_lib.llm.dev.model_adaptation.mllama.convnext_encoder import ConvNeXtCLIPVisionTower
from genai_lib.common.dev.utils import filter_outputs


from transformers.utils import logging
logger = logging.get_logger(__name__)

def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama
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


class QcMllamaTextSelfAttention(MllamaTextSelfAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MllamaTextConfig, layer_idx: int):
        super(QcMllamaTextSelfAttention, self).__init__(config, layer_idx)

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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            if isinstance(position_ids, (tuple, list)): # QC
                position_embeddings  = position_ids
            else:
                position_embeddings = self.rotary_emb(value_states, position_ids)
        cos, sin = position_embeddings
        query_states = _apply_rope_single(query_states, position_embeddings)
        key_states = _apply_rope_single(key_states, position_embeddings)

        # we cache the un-transposed keys before we pass into the combined scoring model.
        untransposed_keys = key_states

        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                            "return_new_key_value_only": return_new_key_value_only,
                            "transposed_key_cache": transposed_key_cache,
                            "num_key_value_heads": self.num_key_value_heads,
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

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QcMllamaTextCrossAttention(MllamaTextCrossAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MllamaTextConfig, layer_idx: int):
        super(QcMllamaTextCrossAttention, self).__init__(config, layer_idx)

    def compute_vision_kv_cache(self, cross_attention_states):
        # ADAPTATION: We allow CrossAttention to compute visionKV cache separately, so this can be made part of VisionEncoder instead of part of LLM
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False
        bsz = cross_attention_states.shape[0]
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        key_states = self.k_norm(key_states)
        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states, value_states = self.compute_vision_kv_cache(cross_attention_states)

            if past_key_value is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position,
                                                               "return_new_key_value_only": return_new_key_value_only,
                                                               "transposed_key_cache": transposed_key_cache,
                                                                "num_key_value_heads": self.num_key_value_heads,
                                                                "head_dim": self.head_dim,}
                )
        else:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )

        if not transposed_key_cache: # Only transpose if NOT done prior to caching
            key_states = key_states.transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def DynamicCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += value_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
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


def DynamicCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.value_cache) <= layer_idx:
        return 0
    return self.value_cache[layer_idx].shape[-2]


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False


orig_causal_mask = modeling_mllama.MllamaTextModel._update_causal_mask
def adapted_update_causal_mask(self, attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    else:
        return orig_causal_mask(self, attention_mask, *args, **kwargs)

orig_embedding_fwd = modeling_mllama.MllamaRotaryEmbedding.forward
def adapted_RotaryEmbedding(self, x, position_ids, *args, **kwargs):
    if isinstance(position_ids, tuple) and len(position_ids)==2:
        return position_ids
    else:
        return orig_embedding_fwd(self, x, position_ids, *args, **kwargs)

class QcMllamaForCausalLM(MllamaForCausalLM):
    """
    Subclass of original LlamaForCausalLM. This is needed to serve two purposes:

    1. Starting from transformers version 4.45.0, the num_logits_to_keep argument is now required argument.
    Consequently, the prepared static graph will always include this additional argument.
    To maintain compatibility with our existing pipelines, we create a new class that inherits from
    LlamaForCausalLM. In this new class, we redefine the forward method without the num_logits_to_keep
    argument and in inside the forward we infer the num_logits_to_keep from the config and then call the superclass's forward method.

    2. For the Long Context scoring model within the LLM, we need to pass two additional arguments:
     anchor_buffer and valid_token_mask. These can be provided as keyword arguments (introduced in transformers version 4.47.0).
     However, this approach is incompatible with Onnx export, as Onnx does not support keyword arguments when creating the onnx graph.
     Therefore, we pass valid_token_mask and anchor_buffer as model inputs to LlamaForCausalLM,
     which in turn get recognized through keyword arguments in the downstream blocks.
      This is similar to how the DynamicCache object is not traced by jit.trace at the topmost level in LlamaForCausalLM.
      """
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference))


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cross_attention_states: Optional[torch.LongTensor] = None,
            cross_attention_mask: Optional[torch.LongTensor] = None,
            full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[torch.Tensor] = None,
            num_logits_to_keep: Optional[int] = None,
            valid_token_mask: Optional[torch.Tensor]=None,
            anchor_buffer: Optional[torch.Tensor]=None,
            cache_index: Optional[torch.Tensor]=None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        num_logits_to_keep = num_logits_to_keep if num_logits_to_keep else getattr(self.config, "num_logits_to_keep", 0)

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcMllamaForCausal doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor

        if type(past_key_values) == tuple:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            position_embeddings = position_embeddings,
            num_logits_to_keep=num_logits_to_keep,
            valid_token_mask=valid_token_mask,
            anchor_buffer=anchor_buffer,
            **kwargs)

        if return_dict:
            assert type(outputs.past_key_values) != tuple
            outputs.past_key_values = outputs.past_key_values.to_legacy_cache()
        else:
            new_outputs = []
            for item in outputs:
                if isinstance(item, DynamicCache):
                    new_outputs.append(item.to_legacy_cache())
                else:
                    new_outputs.append(item)
            outputs = tuple(new_outputs)

        if hasattr(self.config, "output_index_filter"):
            return filter_outputs(outputs, self.config.output_index_filter)
        return outputs


class QcMllamaForConditionalGeneration(MllamaForConditionalGeneration):
    _supports_quantized_cache = False  # quant cache not supported in encoder-decoder setting

    def __init__(self, config: MllamaConfig):
        super().__init__(config)
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference))

        self.vision_model = ConvNeXtCLIPVisionTower(config.vision_config.vision_tower,
                                                    config.vision_config.mm_vision_select_layer,
                                                    config.vision_config.mm_vision_resolution,
                                                    config.vision_config.vision_add_five_stage,
                                                    config.vision_config.vision_five_stage_width)

        self.language_model = QcMllamaForCausalLM._from_config(config.text_config)

        #TODO: FIXME, currently hardcoded, to be fixed by modifying mllamaconfig
        self.select_vision_scale = getattr(config.vision_config,'select_vision_scale', -1)
        if self.select_vision_scale==-1:
            self.vision_output_dim=3072
            self.num_vision_tokens = 144
        elif self.select_vision_scale == -2:
            self.vision_output_dim=int(3072/2)
            self.num_vision_tokens = int(144*4)
        elif self.select_vision_scale == -3:
            self.vision_output_dim=int(3072/4)
            self.num_vision_tokens = int(144*16)

        self.multi_modal_projector_2 = nn.Linear(
            self.vision_output_dim, #TODO: FIXME, currently hardcoded, to be fixed by modifying mllamaconfig
            config.text_config.hidden_size,
            bias=True,
        )
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = None,
        cache_index: Optional[torch.Tensor]=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> prompt = "<|image|>If I had to write a haiku for this one"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> output = model.generate(**inputs, max_new_tokens=15)

        >>> prompt_len = inputs.input_ids.shape[-1]
        >>> generated_ids = output[:, prompt_len:]
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        >>> print(generated_text)
        [', it would be:.\\nA stop sign in Chinatown.\\n']
        ```
        """
        num_logits_to_keep = num_logits_to_keep if num_logits_to_keep else getattr(self.config, "num_logits_to_keep", 0)

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcMllamaForCausal doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        use_convnext = 0
        use_convnext_multi_scale = 1
        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            if use_convnext:  #TODO: FIXME, currently hardcoded, to be fixed by modifying mllamaconfig
                vision_outputs = self.vision_model(pixel_values[:,0,0,...])
            elif use_convnext_multi_scale:  #TODO: FIXME, currently hardcoded, to be fixed by modifying mllamaconfig
                list_image_features = self.vision_model.forward_multiscales(pixel_values[:,0,0,...])
                cross_attention_states = list_image_features[self.select_vision_scale].unsqueeze(1).unsqueeze(1)
                cross_attention_states = self.multi_modal_projector_2(cross_attention_states).reshape(-1, cross_attention_states.shape[-2], self.hidden_size)

            else:
                raise NotImplementedError('---should not use this mode of the original vit')

        if cross_attention_mask is not None:
            if use_convnext or use_convnext_multi_scale: #TODO: FIXME, currently hardcoded, to be fixed by modifying mllamaconfig
                cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                    cross_attention_mask,
                    num_vision_tokens=self.num_vision_tokens,
                    dtype=self.dtype,
                )
            else:
                raise NotImplementedError('---should not use this mode of the original vit')
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        return outputs


def DynamicCache_to_legacy_cache(self):

    """
    Converts the DynamicCache instance to its equivalent in the legacy cache format for backward compatibility.

    The past_key_values passed into the model as input is a tuple.
    The LlamaModel converts it into a Cache object if it isn't one already. Within the model, past_key_values flow as a DynamicCache object.
    Just before returning the output, the LlamaModel converts the DynamicCache back to the legacy cache (tuple format).
    Since we added new attributes to our Cache object, we need to ensure they are included as additional entries in the returned tuple."""

    legacy_cache = ()
    for layer_idx in range(len(self)):
        legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
    if "anchor_buffer" in dir(self):
        return (legacy_cache, self.anchor_buffer)
    return legacy_cache


class QcMllamaCrossAttentionDecoderLayer(MllamaCrossAttentionDecoderLayer):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(self, config: MllamaTextConfig, layer_idx: int):
        super(QcMllamaCrossAttentionDecoderLayer, self).__init__(config, layer_idx)

    def _compute_tanh(self):
        # Workaround for Converter failure on Tanh buffer
        logger.info("Pre-Computed Tanh(s)")
        cross_attn_attn_gate_tanh = self.cross_attn_attn_gate.tanh().detach()
        cross_attn_mlp_gate_tanh = self.cross_attn_mlp_gate.tanh().detach()
        self.register_buffer('cross_attn_attn_gate_tanh', cross_attn_attn_gate_tanh) # MPP fix
        self.register_buffer('cross_attn_mlp_gate_tanh', cross_attn_mlp_gate_tanh) # MPP fix

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, past_key_value = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

        if hasattr(self, 'cross_attn_attn_gate_tanh'): # Workaround for Converter failure on Tanh buffer
            hidden_states = residual + self.cross_attn_attn_gate_tanh * hidden_states
        else:
            hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore

        if hasattr(self, 'cross_attn_mlp_gate_tanh'): # Workaround for Converter failure on Tanh buffer
            hidden_states = residual + self.cross_attn_mlp_gate_tanh * hidden_states
        else:
            hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs
