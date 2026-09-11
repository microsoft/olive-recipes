
#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from abc import ABC
from typing import List, Optional, Tuple, Union
import aimet_torch.v2.nn.modules.custom as aimet_ops
import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward
from genai_lib.common.dev.utils import filter_outputs
from transformers.utils import logging

logger = logging.get_logger(__name__)

__all__ = ["BaseDraftModel"]

class BaseDraftModel(nn.Module, ABC):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`EagletDecoderLayer`]
    """

    def __init__(self, config, decoder_cls, norm_cls=None, dual_fc=False):
        """
        Args:
            config: pretrained configuration for the model.
                - dual_fc: whether to use dual fc layers to combine input embeddings and hidden states.
                - fce_bias: whether to use bias in fce layer when dual_fc is True.
                - fch_bias: whether to use bias in fch layer when dual_fc is True
            decoder_cls: decoder class.
                decoder layer's forward is expected to return a tensor `hidden_states` or a tuple of
                (hidden_states, attention_weights, next_cache) where:
                - `hidden_states` is the output of the layer
                - [Optional] `attention_weights` exists if `output_attentions` is True else not in the tuple
                - [Optional] `next_cache` exists if `use_cache` is True else not in the tuple
            norm_cls: normalization layer class for final layer norm. If None, no final layer norm is applied.
        Raises:
            TypeError: if config or decoder_cls are not of the correct type.
        """
        super().__init__()
        dual_fc = getattr(config, "dual_fc", dual_fc)
        self.config = config
        # Support downsample_size for backward compatibility
        self.bottleneck_size = getattr(config, "downsample_size", None) or getattr(config, "bottleneck_size", None)
        if self.bottleneck_size is None:
            raise ValueError(
                "Configuration error: Please check the config and make sure 'bottleneck_size' is set in the config."
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.gradient_checkpointing = False
        self.training = False
        self.layers = nn.ModuleList(
            [decoder_cls(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if dual_fc:
            self.fce = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, "fce_bias", False))
            self.fch = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, "fch_bias", False))
        else:
            self.embedding_concat = aimet_ops.Concat(axis=-1)
            self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=config.fc_bias)
        self.down_proj = nn.Linear(config.hidden_size, self.bottleneck_size, bias=False)
        self.up_proj = nn.Linear(self.bottleneck_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.merged_head = nn.Linear(self.bottleneck_size, getattr(config, "trimmed_vocabulary_length", config.vocab_size), bias=False)
        if norm_cls is not None:
            self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference), persistent=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward('Draft model forward')
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        hidden_states: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None,
        cache_index: Optional[torch.Tensor]=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), f"{self.__class__.__name__} doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor

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
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # QC Adaptation: Assign pre-computed position embeddings if available
        if isinstance(position_ids, (tuple, list)) and len(position_ids) == 2:
            position_embeddings = position_ids
        else:
            raise ValueError("`position_ids` should be a pre-computed position embedding.")

        # QC Adaptation: Assign pre-computed causal mask if available
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            raise ValueError("Please provide a pre-computed 4D causal mask within `attention_mask`.")

        # embed positions
        if hasattr(self, "fce") and hasattr(self, "fch"):
            hidden_states = self.fce(inputs_embeds) + self.fch(hidden_states)
        else:
            concat_embeddings = self.embedding_concat(inputs_embeds, hidden_states)
            hidden_states = self.fc(concat_embeddings)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # hidden_up
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.act(hidden_states)

        # add hidden states from the down proj
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.merged_head(hidden_states[:, slice_indices, :])

        if not return_dict:
            outputs = tuple(v for v in [None, logits, next_cache, all_hidden_states, None] if v is not None)
            if hasattr(self.config, "output_index_filter"):
                outputs = filter_outputs(outputs, self.config.output_index_filter)
            return outputs

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
