#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
#
# This file contains certain notices of software components included with the
# software that Qualcomm Technologies, Inc. ("QTI") is required to provide you.
# Except where prohibited by the open source license, the content of this file is
# provided solely to satisfy QTI's attribution and notice requirement; your use of
# these software components together with the QTI software ("Software") is subject
# to the terms of your license from QTI. Compliance with all copyright laws and
# software license agreements included in the notice section of this file are the
# responsibility of the user. Except as may be granted by separate express written
# agreement, this file provides no license to any patents, trademarks, copyrights,
# or other intellectual property of Qualcomm Incorporated or any of its
# subsidiaries.
#
# Software provided with this notice is NOT A CONTRIBUTION to any open source
# project. If alternative licensing is available for any of the components with
# licenses or attributions provided below, a license choice is made for receiving
# such code by QTI.

# Copyright (c) 2023 Qualcomm Technologies, Inc. All rights reserved.

# Qualcomm is a trademark of Qualcomm Incorporated, registered in the United
# States and other countries. All Qualcomm Incorporated trademarks are used with
# permission. Other products and brand names may be trademarks or registered
# trademarks of their respective owners.
#
# =============================================================================
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
#
# =============================================================================

"""utility method to adapt original model, prepared model and model forward pass invocation"""

import inspect

import contextlib

import json
import math
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from aimet_torch.utils import get_device

from packaging import version
from importlib.metadata import version as impLib_version

from transformers.models.phi3.modeling_phi3 import DynamicCache


class Phi4RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        partial_rotary_factor=0.75,
        device=None,
    ):
        super().__init__()

        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor
        self.dim = int(dim * self.partial_rotary_factor)
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(
                        0, self.dim, 2, dtype=torch.int64, device=x.device
                    ).float()
                    / self.dim
                )
            )
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi4LongRoPEScaledRotaryEmbedding(Phi4RotaryEmbedding):
    def __init__(self, dim, config, partial_rotary_factor, device=None):
        super().__init__(
            dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            device=device,
        )

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = seq_len or torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(
                self.long_factor, dtype=torch.float32, device=x.device
            )
        else:
            ext_factors = torch.tensor(
                self.short_factor, dtype=torch.float32, device=x.device
            )

        inv_freq_shape = (
            torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float()
            / self.dim
        )

        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(
                    1
                    + math.log(scale) / math.log(self.original_max_position_embeddings)
                )

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def flatten_tensors(tup):
    if not isinstance(tup, (tuple, list)):
        yield tup
        return
    for x in tup:
        yield from flatten_tensors(x)


def get_padded_kv_values(
    past_size,
    num_layers,
    hidden_size,
    num_attention_heads,
    batch_size=1,
    num_kv_heads=32,
    transposed_key_cache=True,
    device="cuda",
    dtype=torch.float32,
):
    def _cache(shape):
        return torch.zeros(shape).to(device=device, dtype=dtype)

    head_dim = num_kv_heads
    value = (batch_size, head_dim, past_size, hidden_size // num_attention_heads)
    key = (
        (value[0], value[1], value[3], value[2])
        if transposed_key_cache
        else tuple(value)
    )
    past_key_values = tuple((_cache(key), _cache(value)) for _ in range(num_layers))
    return past_key_values


class RopeEmbedding:
    def __init__(
        self,
        device,
        head_dim=128,
        max_length=2048,
        partial_rotary_factor=0.75,
        config=None,
    ):
        self.cos, self.sin = self.precompute(
            device, head_dim, partial_rotary_factor, max_length, config
        )

    def precompute(self, device, head_dim, partial_rotary_factor, max_length, config):
        def _support_llama3_rope():
            import transformers

            return tuple([int(i) for i in transformers.__version__.split(".")]) >= (
                4,
                43,
                2,
            )
            # return version.parse(impLib_version('transformers')) >= version.parse('4.43.2')

        head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        kwargs = {
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
            "device": device,
        }
        if _support_llama3_rope():
            kwargs["config"] = config

        if not hasattr(config, "rope_scaling"):
            setattr(config, "rope_scaling", None)

        if config.rope_scaling is None:
            rope = Phi4RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                partial_rotary_factor=partial_rotary_factor,
                device=device,
            )
        else:
            scaling_type = config.rope_scaling["type"]
            if scaling_type == "longrope":
                rope = Phi4LongRoPEScaledRotaryEmbedding(
                    dim=head_dim,
                    config=config,
                    partial_rotary_factor=partial_rotary_factor,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        dummy_x = torch.Tensor([1.0]).to(device)
        # dummy_x device
        position_ids = torch.arange(max_length).view(1, -1).to(device)
        if hasattr(rope, "_original_forward"):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        embeddings = [emb.unsqueeze(0) for emb in embeddings]
        return embeddings

    def get_embedding(self, position_ids, dtype=torch.float32):
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :]  # [seq_len, dim]
        sin = self.sin[0, 0, :, :]  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


def prepare_decoder_attention_mask(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    mask_neg=-100.0,
    sliding_window=None,
):
    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        mask_neg: float = -100.0,
        sliding_window=None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape[0], input_ids_shape[1]
        # mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask = torch.full(
            (tgt_len, tgt_len), torch.tensor(mask_neg, device=device), device=device
        )
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.zeros(
                        tgt_len, past_key_values_length, dtype=dtype, device=device
                    ),
                    mask,
                ],
                dim=-1,
            )

        # sliding window attention
        if sliding_window is not None:
            sliding_attn_mask = torch.zeros(mask.shape, device=device)
            sliding_mask_cond = (
                mask_cond + past_key_values_length - sliding_window
            ).view(mask_cond.size(0), 1)
            sliding_attn_mask.masked_fill_(
                torch.arange(tgt_len + past_key_values_length, device=device)
                <= sliding_mask_cond,
                mask_neg,
            )
            mask += sliding_attn_mask

        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(
        mask: torch.Tensor,
        dtype: torch.dtype,
        mask_neg: float = -100.0,
        tgt_len: int = None,
    ):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = (
            mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        )

        inverted_mask = 1.0 - expanded_mask

        # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), mask_neg)

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            mask_neg=mask_neg,
            sliding_window=sliding_window,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        expanded_attn_mask = _expand_mask(
            attention_mask,
            inputs_embeds.dtype,
            tgt_len=input_shape[1],
            mask_neg=mask_neg,
        ).to(inputs_embeds.device)

        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def get_position_embeddings_from_position_ids(
    position_ids, head_dim, max_length, partial_rotary_factor, device, dtype, config
):
    return RopeEmbedding(
        device=device,
        head_dim=head_dim,
        max_length=max_length,
        partial_rotary_factor=partial_rotary_factor,
        config=config,
    ).get_embedding(position_ids, dtype=dtype)


def prepare_combined_attention_mask(
    attention_mask,
    input_shape,
    past_key_values_length,
    device,
    mask_neg=-100.0,
    sliding_window=None,
    dtype=torch.float32,
):
    dummy_embedding = torch.tensor((1.0,)).to(torch.float32).to(device)
    new_mask = prepare_decoder_attention_mask(
        attention_mask,
        input_shape,
        dummy_embedding,
        past_key_values_length,
        mask_neg,
        sliding_window,
    )
    return new_mask.clamp_min(mask_neg).to(dtype)


class LLMForwardPassManager:
    def __init__(self, cfg, model, tokenizer, separate_tuple_input_output, num_tokens):
        self.tokenizer = tokenizer
        self.model = model
        self.config = cfg
        self.device = get_device(model)

        self.num_heads = getattr(cfg, "num_attention_heads", 1)
        self.num_kv_heads = getattr(cfg, "num_key_value_heads")
        self.num_layers = getattr(cfg, "num_hidden_layers", 32)
        self.embed_dim = getattr(cfg, "hidden_size", 1024)
        self.rope_theta = getattr(cfg, "rope_theta", 10000.0)
        self.sliding_window = getattr(cfg, "sliding_window", None)
        self.max_tokens = tokenizer.model_max_length
        self.num_tokens = num_tokens
        self.use_position_embedding_input = getattr(
            cfg, "use_position_embedding_input", False
        )
        self.use_combined_mask_input = getattr(cfg, "use_combined_mask_input", False)
        self.transposed_key_cache = getattr(cfg, "transposed_key_cache", False)
        self.mask_neg = getattr(cfg, "mask_neg", -100)
        self.use_input_embeddings = getattr(cfg, "use_input_embeddings", False)
        self.return_new_key_value_only = getattr(
            cfg, "return_new_key_value_only", False
        )
        self.use_cache = getattr(cfg, "use_cache", False)
        self.separate_tuple_input_output = separate_tuple_input_output
        self.record_test_vectors = False  # users of this block wil enable/disable this as necessary with provided functions
        self.dummy_kvcache_generator = None  # DummyKvcacheGenerator(cfg)
        self.input_id_to_embedding_converter = None
        self.partial_rotary_factor = getattr(cfg, "partial_rotary_factor", 0.75)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def replace_model(self, new_model):
        self.model = new_model
        self.model.to(self.device)

    @contextlib.contextmanager
    def place_on_device(self, device):
        original_device = self.device
        try:
            self.to(device)
            yield
        finally:
            self.to(original_device)

    def to(self, device=torch.device):
        self.device = torch.device(device)
        self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def _tokenize_text(self, text, max_length):
        if self.tokenizer == None:
            print(
                "No tokenizer was registered with forward pass manager. Attempt to forward text inputs has failed."
            )
            assert False

        encoded_tensor = self.tokenizer(
            text, add_special_tokens=False, max_length=max_length, truncation=True
        )
        return encoded_tensor

    def _update_kv_cache(
        self, prev_key_value, new_key_value, max_cache_size, is_concatenated=False
    ):
        # past_key_value: [num_layers][2][key_value], where key_value can be a tensor or tuple of heads
        def _concat(a, b, dim):
            if isinstance(a, tuple):
                assert len(a) == len(b), "Unexpected key/value pair"
                return tuple(_concat(ai, bi, dim) for ai, bi in zip(a, b))
            return torch.cat((a, b), dim=dim)

        def _do_concat(a, b, key_dim, value_dim):
            return tuple(
                (_concat(ak, bk, key_dim), _concat(av, bv, value_dim))
                for (ak, av), (bk, bv) in zip(a, b)
            )

        def _shift(a, dim, shift_size):
            if isinstance(a, tuple):
                return tuple(_shift(ai, dim) for ai in a)
            assert dim in (2, 3), "Unexpected shift axis"
            return a[:, :, shift_size:, :] if dim == 2 else a[:, :, :, shift_size:]

        def _do_shift(a, key_dim, value_dim, shift_size):
            return tuple(
                (_shift(k, key_dim, shift_size), _shift(v, value_dim, shift_size))
                for k, v in a
            )

        value_dim = 2
        key_dim = 3 if self.transposed_key_cache else 2

        if prev_key_value is None or is_concatenated:
            # some models concat new key values and old key values internally
            # `is_concatenated` indicates whether new_key_value is already concatenated
            next_key_value = new_key_value
        elif new_key_value is None:
            # when dummy_kv + None
            next_key_value = prev_key_value
        else:
            # if concat is NOT done, then concat
            next_key_value = _do_concat(
                prev_key_value, new_key_value, key_dim, value_dim
            )

        shift_size = next_key_value[0][1].shape[-2] - max_cache_size
        if shift_size > 0:
            next_key_value = _do_shift(next_key_value, key_dim, value_dim, shift_size)

        return next_key_value

    def validate_inputs(
        self,
        input_text=None,
        input_ids=None,
        input_embeddings=None,
        past_key_values=None,
    ):
        # make sure only one of input_text, input_ids, input_embeddings is passed in
        input_count = 0
        for input in (input_text, input_ids, input_embeddings):
            if input is not None:
                input_count = input_count + 1
        if input_count != 1:
            print(
                "Incorrect number of arguments: one of (input_text, input_ids, input_embeddings) expected."
            )
            return False

        # make sure that input embedding function has been selected if input embeddings are to be used
        if (
            self.use_input_embeddings
            and self.input_id_to_embedding_converter is None
            and input_embeddings is None
        ):
            print(
                "use_input_embeddings is set to true, but no input_embeddings were provided, and input_id_to_embedding_converter is None."
            )
            return False

        if (
            past_key_values is not None
            and past_key_values[0][1].shape[-2] > self.max_tokens - self.num_tokens
        ):
            print(
                "Provided past_key_values are too long. past_key_values length cannot exceed max_tokens - num_tokens."
            )
            return False

        return True

    def validate_input_lengths(self, input_length, mask_length, attn_length):
        if 1 > input_length or input_length > self.num_tokens:
            print(
                f"Incorrect sequence length provided: input_length({input_length}) must be less than or equal to num_tokens ({self.num_tokens})."
            )
            return False

        if attn_length < mask_length or mask_length < input_length:
            print(
                f"Incorrect attention length provided: mask_length({mask_length}) must be greater than or equal to input_length({input_length}) and less than or equal to the sum({attn_length}) of input_length and kv_length."
            )
            return False

        return True

    def validate_processed_inputs(
        self, input=None, attention_mask=None, past_key_values=None
    ):
        # if input make sure that only correct length sequence is provided
        if input.shape[1] != self.num_tokens:
            print(
                f"Incorrect prcessing for sequence length: dim 1({input.shape[1]}) of input must be of length num_tokens in KV cache mode."
            )
            return False

        if attention_mask.shape[1] != self.max_tokens:
            print(
                f"Incorrect prcessing for attention length: dim 1({attention_mask.shape[1]}) of input must be of length max_tokens."
            )
            return False

        if (
            past_key_values is not None
            and past_key_values[0][1].shape[-2] != self.max_tokens - self.num_tokens
        ):
            print(
                f"Incorrect  prcessing for past_kv length: dim 1({past_key_values[0][1].shape[-2]}) of input must be of length max_tokens - num_tokens."
            )
            return False

        return True

    def get_position_embeddings_from_position_ids(self, position_ids):
        return get_position_embeddings_from_position_ids(
            position_ids,
            head_dim=self.embed_dim // self.num_heads,
            max_length=self.max_tokens,
            partial_rotary_factor=self.partial_rotary_factor,
            device=self.device,
            dtype=self.dtype,
            config=self.config,
        )

    def prepare_combined_attention_mask(
        self, attention_mask, input_shape, past_kv_length, sliding_window
    ):
        return prepare_combined_attention_mask(
            attention_mask,
            input_shape=input_shape,
            past_key_values_length=past_kv_length,
            device=self.device,
            mask_neg=self.mask_neg,
            sliding_window=sliding_window,
            dtype=self.dtype,
        )

    def prepare_inputs(
        self,
        input_text=None,
        input_ids=None,
        input_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        assert self.validate_inputs(
            input_text, input_ids, input_embeddings, past_key_values
        )

        kvcache_info_bundle = {}  # dict to hold values needed for KV cache post-processing

        if input_text is not None:
            max_length = self.num_tokens
            encoded = self._tokenize_text(input_text, max_length=max_length)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask

        if self.use_input_embeddings:
            if input_embeddings is None:
                input_embeddings = self.input_id_to_embedding_converter(input_ids).to(
                    dtype=self.dtype
                )
            input = input_embeddings
            # if we cast this input to long, all floats become zero in the input which we do not want
            input = torch.tensor(
                input.clone().detach(), dtype=self.dtype, device=self.device
            )
        else:
            input = input_ids
            input = torch.tensor(
                input.clone().detach(), dtype=torch.long, device=self.device
            )
        batch_size = input.shape[0]
        input_length = input.shape[1]

        kvcache_info_bundle["input_length"] = input_length

        # get kv_length from past values because values are not transposed.
        kv_length = (
            past_key_values[0][1].shape[-2] if past_key_values is not None else 0
        )
        attn_length = min(input_length + kv_length, self.max_tokens)

        # Checking attention_mask first, otherwise we will create attention_mask from input_extensions.
        # input_extensions will be empty tensors and so as attention_mask.
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_length + kv_length),
                dtype=torch.long,
                device=self.device,
            )

        # cast type and move device
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(dtype=torch.long, device=self.device)
        else:
            # if attention_mask is not a tensor, get tensor
            attention_mask = torch.tensor(
                attention_mask, dtype=torch.long, device=self.device
            )
        mask_length = attention_mask.shape[1]

        assert self.validate_input_lengths(input_length, mask_length, attn_length)

        # Pad inputs
        if input_length < self.num_tokens:
            shape = (batch_size, self.num_tokens - input_length)
            # expand shape if input is input_embeddings
            if self.use_input_embeddings:
                shape += (input.shape[-1],)
            input_extensions = torch.full(
                shape,
                fill_value=self.tokenizer.eos_token_id,
                dtype=input.dtype,
                device=self.device,
            )
            input = torch.cat((input_extensions, input), dim=1)

        # Pad attention_mask
        attention_mask_extension_for_padded_kvcache = torch.zeros(
            (batch_size, attn_length - mask_length),
            dtype=torch.long,
            device=self.device,
        )
        attn_mask_extensions_for_padded_input = torch.zeros(
            (batch_size, self.num_tokens - input_length),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.cat(
            (
                attention_mask_extension_for_padded_kvcache,
                attention_mask[:, :-input_length],
                attn_mask_extensions_for_padded_input,
                attention_mask[:, -input_length:],
            ),
            dim=1,
        )

        desired_kv_length = self.max_tokens - self.num_tokens
        kv_padding_length = max(desired_kv_length - kv_length, 0)
        kvcache_info_bundle["kv_padding_length"] = kv_padding_length

        past_key_values_extension = get_padded_kv_values(
            past_size=kv_padding_length,
            num_layers=self.num_layers,
            hidden_size=self.embed_dim,
            num_attention_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transposed_key_cache=self.transposed_key_cache,
            device=self.device,
            dtype=self.dtype,
        )
        past_key_values = self._update_kv_cache(
            past_key_values_extension, past_key_values, desired_kv_length
        )

        attention_mask_extension = torch.zeros(
            (batch_size, kv_padding_length), dtype=torch.long, device=self.device
        )
        attention_mask = torch.cat((attention_mask_extension, attention_mask), dim=1)

        assert self.validate_processed_inputs(input, attention_mask, past_key_values)

        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids = position_ids.clip(0, self.max_tokens - 1)
        position_ids = position_ids[..., -self.num_tokens :]

        if self.use_position_embedding_input:
            position_ids = self.get_position_embeddings_from_position_ids(position_ids)

        if self.use_combined_mask_input:
            past_kv_length = self.max_tokens - self.num_tokens
            attention_mask = self.prepare_combined_attention_mask(
                attention_mask, input.shape, past_kv_length, self.sliding_window
            )

        inputs = {
            "attention_mask": attention_mask,
        }

        if (
            self.separate_tuple_input_output
            and self.config.use_position_embedding_input
        ):
            inputs["position_ids_cos"] = position_ids[0]
            inputs["position_ids_sin"] = position_ids[1]
        else:
            inputs["position_ids"] = position_ids

        if self.use_input_embeddings:
            inputs["inputs_embeds"] = input
        else:
            inputs["input_ids"] = input

        if self.separate_tuple_input_output:
            if "input_names" in kwargs:
                input_names = kwargs["input_names"]
            else:
                signature = inspect.signature(self.model.forward)
                input_names = tuple(signature.parameters.keys())
            flattened_key_values = flatten_tensors(past_key_values)
            # input_ids, attention_mask, position_ids_cos, position_ids_sin, (past_key_values)
            # this order is different when we use the input_embeddings -> attention_mask, position_ids_cos, position_ids_sin, (past_key_values), inputs_embeds
            if not self.use_input_embeddings:
                offset = 4 if self.config.use_position_embedding_input else 3
                for key, value in zip(input_names[offset:], flattened_key_values):
                    inputs[key] = value
            else:
                for key, value in zip(input_names[3:-1], flattened_key_values):
                    inputs[key] = value
        else:
            inputs["past_key_values"] = past_key_values

        return inputs, kvcache_info_bundle

    def prepare_outputs(self, outputs, prepared_inputs, kvcache_info_bundle):
        """
        Args:
            outputs (tuple): Tuple of model outputs.
                outputs[0]: logits (batch, num_tokens, vocab_size)
                outputs[-1]: kv caches with max_tokens length
            prepared_inputs (dict): Dictionary of prepared inputs.
            kvcache_info_bundle (dict): Dictionary containing information about key-value cache.

        Returns:
            dict: A dictionary containing 'lm_logits' and 'past_key_values'.
                lm_logits: (batch, num_tokens, vocab_size)
                past_key_values: having length as the number of non-dummy inputs
        """
        lm_logits = outputs[0]
        lm_logits = lm_logits[:, -kvcache_info_bundle["input_length"] :, :]

        def _get_past_kv_from_outputs(outputs):
            if self.separate_tuple_input_output:
                return tuple(
                    (outputs[(2 * i) + 1], outputs[(2 * i) + 2])
                    for i in range(self.num_layers)
                )
            else:
                return outputs[-1]

        def _get_past_kv_from_prepared_inputs(prepared_inputs):
            if self.separate_tuple_input_output:
                return tuple(
                    (
                        prepared_inputs[f"past_key_{i}_in"],
                        prepared_inputs[f"past_value_{i}_in"],
                    )
                    for i in range(self.num_layers)
                )
            else:
                return (
                    prepared_inputs["past_key_values"]
                    if "past_key_values" in prepared_inputs
                    else None
                )

        new_past_key_values = _get_past_kv_from_outputs(outputs)
        new_past_key_values = self._update_kv_cache(
            None, new_past_key_values, kvcache_info_bundle["input_length"]
        )
        old_past_key_values = _get_past_kv_from_prepared_inputs(prepared_inputs)

        current_kv_length_with_padding_removed = (
            self.max_tokens
            - self.num_tokens
            - kvcache_info_bundle["kv_padding_length"]
            + kvcache_info_bundle["input_length"]
        )  # number of non-dummy inputs

        past_key_values = self._update_kv_cache(
            old_past_key_values,
            new_past_key_values,
            current_kv_length_with_padding_removed,
        )

        return {"lm_logits": lm_logits, "past_key_values": past_key_values}

    def __call__(self, *args, **kwargs):
        prepared_inputs, kvcache_info_bundle = self.prepare_inputs(*args, **kwargs)
        outputs = self.model(**prepared_inputs)
        prepared_outputs = self.prepare_outputs(
            outputs, prepared_inputs, kvcache_info_bundle
        )
        return prepared_outputs


def slice_inputs_and_run_successive_kvcache_inference(
    fpm, input_ids=None, input_embeds=None, **kwargs
):
    if input_ids is not None:
        input_length = input_ids.shape[1]
    else:
        input_length = input_embeds.shape[1]

    outputs = {}

    attention_mask = kwargs.pop("attention_mask", None)

    for idx in range(0, input_length, fpm.num_tokens)[::-1]:
        idx = input_length - idx

        if attention_mask is not None:
            cache_offset = attention_mask.shape[1] - input_length
            kwargs["attention_mask"] = attention_mask[
                :, max(0, cache_offset + idx - fpm.max_tokens) : cache_offset + idx
            ]

        if input_ids is not None:
            cur_outputs = fpm(
                input_ids=input_ids[:, max(0, idx - fpm.num_tokens) : idx], **kwargs
            )
        elif input_embeds is not None:
            cur_outputs = fpm(
                input_ids=None,
                input_embeddings=input_embeds[:, max(0, idx - fpm.num_tokens) : idx, :],
                **kwargs,
            )
        else:
            print("No input_ids or inputs_embeds provided to inference generator!")
            assert False

        # get valid outputs
        bsz, length, dim = cur_outputs["lm_logits"].shape

        outputs["lm_logits"] = torch.cat(
            (
                outputs.get("lm_logits", torch.zeros((bsz, 0, dim), device=fpm.device)),
                cur_outputs["lm_logits"],
            ),
            dim=1,
        )
        kwargs["past_key_values"] = outputs["past_key_values"] = cur_outputs[
            "past_key_values"
        ]

    return outputs
