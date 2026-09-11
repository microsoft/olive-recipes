#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch
from genai_lib.llm.utils import _shift, _concat, _get_position_emb_names
from genai_lib.llm.dev.model_adaptation.common.utils import KEY_CONCAT_AXIS, VALUE_CONCAT_AXIS
import warnings

def lfm2_get_dummy_kv(batch_size,num_key_value_heads, head_dim, conv_dim, conv_len, key_concat_axis, device, dtype=torch.float32, cache_len = None, model_context_len=None, max_input_tokens=None, enable_shortconv_native=False):
    """
    This function determines the shape of the dummy kv and conv cache using the required arguments which reflect model config
    Returns the dummy kv of fixed size each time (for a single layer). This will be used for padding the passed past kv

    params:
    batch_size: the batch size needed to create dummy kv
    model_context_len : model_context_len: maximum number of tokens that the model can consume in total
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    num_key_value_heads: the number of key value heads
    head_dim: dimension at each head
    key_concat_axis: the axis to which we want to append the keys
    device: the device to place dummy kv on, this is inferred from the unpadded_past_kv tensor if it is not None
    enable_shortconv_native: if use shortconv native, the cache should be expand into 4D instead of 3D
    """

    def _cache(shape):
        return torch.zeros(shape, device=device, dtype=dtype)

    if cache_len is None:
        cache_len = model_context_len-max_input_tokens

    value = (batch_size, num_key_value_heads,cache_len , head_dim)
    key = (value[0], value[1], value[3], value[2]) if key_concat_axis == 3 else tuple(value)
    conv = (batch_size, conv_dim, conv_len)
    if enable_shortconv_native:
        conv = (batch_size, conv_dim, 1, conv_len)
    return (_cache(key), _cache(value), _cache(conv))

def lfm2_pad_past_kv(attention_indices, dummy_past_kv, unpadded_past_kv, num_hidden_layers, key_concat_axis, value_concat_axis=2, pad_to_left=True):
    """
    This function is responsible taking in current past kv and pad it using dummy kv to meet the static shape
    requirements for past kv.
    We compute the padding kv length as (Context Length - AR length) - (valid kv length).
    The shape after we pad past kv is (Context Length - AR length)

    params:
    dummy_past_kv: this corresponds to the dummy kv for one hidden layer, it is same for all the layers or a list for where each entry is a tuple, which is the dummy kv for the particular layer
    unpadded_past_kv: this is the useful accumulated past kv (require this to obtain the length of useful past kv)
    num_hidden_layers: The number of decoder blocks in the model
    key_concat_axis: the axis to which we want to append the keys
    value_concat_axis: the axis to which we want to append the values
    pad_to_left: boolean value indicating whether padding is done towards the left or right.

    """

    if isinstance(dummy_past_kv, list):
        raise NotImplementedError("Not support dummy past kv list")

    else:
        first_attention_idx = attention_indices[0]
        useful_past_kv_length = unpadded_past_kv[first_attention_idx][1].shape[-2] if unpadded_past_kv else 0

        # trimmed dummy kv is the final length dummy kv that will be concatenated to the unpadded_past_kv either to the left or to the right.
        trimmed_dummy_kv = (_shift(dummy_past_kv[0], key_concat_axis, useful_past_kv_length), _shift(dummy_past_kv[1], value_concat_axis, useful_past_kv_length))
        if unpadded_past_kv:
            if pad_to_left:
                padded_key_values = []
                for i in range(num_hidden_layers):
                    if i in attention_indices:
                        key = _concat(trimmed_dummy_kv[0], unpadded_past_kv[i][0], key_concat_axis)
                        value = _concat(trimmed_dummy_kv[1], unpadded_past_kv[i][1], value_concat_axis)
                        padded_key_values.append((key, value))
                    else:
                        conv = dummy_past_kv[2]
                        if unpadded_past_kv[i] is not None:
                            conv = unpadded_past_kv[i]
                        padded_key_values.append(conv)
            else:
                padded_key_values = []
                for i in range(num_hidden_layers):
                    if i in attention_indices:
                        key = _concat(unpadded_past_kv[i][0], trimmed_dummy_kv[0], key_concat_axis)
                        value = _concat(unpadded_past_kv[i][1], trimmed_dummy_kv[1], value_concat_axis)
                        padded_key_values.append((key, value))
                    else:
                        conv = dummy_past_kv[2]
                        if unpadded_past_kv[i] is not None:
                            conv = unpadded_past_kv[i]
                        padded_key_values.append(conv)

            return tuple(padded_key_values)
        else:
            padded_key_values = []
            for i in range(num_hidden_layers):
                if i in attention_indices:
                    padded_key_values.append(trimmed_dummy_kv)
                else:
                    conv = dummy_past_kv[2]
                    padded_key_values.append(conv)
            return tuple(padded_key_values)

def _get_past_key_value_names(attention_indices, sfx, n_layers, separate_tuple_input_output):
    if not separate_tuple_input_output:
        return ["past_key_values"]
    all = []
    for i in range(n_layers):
        if i in attention_indices:
            all.append(f'past_key_{i}_{sfx}')
            all.append(f'past_value_{i}_{sfx}')
        else:
            all.append(f'cache_conv_{i}_{sfx}')

    return all

def lfm2_model_input_output_names(attention_indices, num_hidden_layers, use_position_embedding_input=True , separate_tuple_input_output=False, use_input_embedding = False):
    '''
    This function is responsible for returning a list of the model input and output names based on the number of hidden layers for LFM2 signature of inputs
    params:
    num_hidden_layers: number of hidden layers of the model
    use_position_embedding_input: are the position ids supplied to the model in embeddings form (assume sin and cos embedding, if yes)
    separate_tuple_input_output: are the inputs passed into the model in tupled format or not
    use_input_embedding: do we pass input ids or input embeddings
    '''
    input_names=['input_ids', 'attention_mask', 'conv_mask']
    input_names += _get_position_emb_names(use_position_embedding_input=use_position_embedding_input, separate_tuple_input_output=separate_tuple_input_output)
    output_names = ['logits']
    input_names += _get_past_key_value_names(attention_indices, "in", num_hidden_layers, separate_tuple_input_output)
    output_names += _get_past_key_value_names(attention_indices, "out", num_hidden_layers, separate_tuple_input_output)
    if use_input_embedding:
        input_names += ['inputs_embeds']
        input_names.pop(0)
    return input_names, output_names

def lfm2_update_kv_cache(unpadded_past_kv, current_key_values,  key_concat_axis=KEY_CONCAT_AXIS, value_concat_axis=VALUE_CONCAT_AXIS, input_ids_slice = None, inputs_embeds_slice=None, pad_to_left=True, skip_pad_layers=(), attention_layer_ids=None):
    """
    This function concats the KV cache that the model outputs in the current iteration (unpadded_past_kv) with the KV$ that the model has accumulated so far(unpadded_past_kv)
    1. remove the non-useful padding kv from the current_key_values depending on whether it was padded to left or to right
    2. concatenate the stripped current kv with past useful kv if it exists
    3. Handle conv cache (should be passthrough since model took care of it)

    params:
    1. unpadded_past_kv: the unpadded useful kv that is accumulated from the previous model invokations
    2. current_key_values: current padded kv returned from the model
    3. key_concat_axis: the axis to which we want to append the keys
    4. value_concat_axis: the axis to which we want to append the values
    5. input_ids_slice: the slice of inputs returned from the iterator (this is before any padding has been applied to meet the static shape requirement)
    6. inputs_embeds_slice: the slice of inputs returned from the iterator (this is before any padding has been applied to meet the static shape requirement)
    7. pad_to_left: boolean value indicating whether padding is done towards the left or right.
    8. skip_pad_layers: Layers not to pad, such as layers which are not updated each LLM inference (e.g. cross-attention layers)
    9. attention_layer_ids: Layers of attention, used to separate between attention layer and short conv layer

    """
    input = input_ids_slice if input_ids_slice is not None else inputs_embeds_slice

    # TODO determine whether we need to trim or not based on the current_pad_len, if negative do not trim. min set to 0
    trimmed_current_key_values = trim_current_kv(
        current_key_values,
        input,
        key_concat_axis,
        value_concat_axis,
        pad_to_left,
        layer_indices_to_perform_trimming=attention_layer_ids,
    )
    # slicing in place before sending to concat function to avoid  memory spiking.
    if unpadded_past_kv:
        concatenated_key_values = []
        for i, (unpadded_cache, current_cache) in enumerate(zip(unpadded_past_kv, trimmed_current_key_values)):
            if i in attention_layer_ids:
                (unpadded_key, unpadded_value) = unpadded_cache
                (current_key, current_value) = current_cache
                if i not in skip_pad_layers:
                    key = _concat(unpadded_key, current_key, key_concat_axis)
                    value = _concat(unpadded_value, current_value, value_concat_axis)
                else:
                    key = current_key_values[i][0]
                    value = current_key_values[i][1]
                concatenated_key_values.append((key, value))
            else:
                concatenated_key_values.append(current_cache)

        concatenated_key_values = tuple(concatenated_key_values)

        return concatenated_key_values

    return trimmed_current_key_values

def trim_current_kv(current_key_values, input,  key_concat_axis, value_concat_axis=2, trim_from_left=True, layer_indices_to_perform_trimming = None):
    """
    params:
    1. current_key_values: current padded/ unpadded kv returned from the model
    2. input: tensor with the shape (at dimension 1) of the post trimmed KV$
    3. key_concat_axis: the axis to which we want to append the keys
    4. value_concat_axis: the axis to which we want to append the values
    5. trim_from_left: whether to trim from left or right
    6. layer_idx_to_perform_trimming: None if trimming to be done to all layers, else pass a list representing indices on which we want to perform trimming.
    if the current_pad_length we compute is positive, means the current keys have padding which need to be removed. But if the current_pad_length is negative, we do not remove anything since the user has already trimmed the kv before. (This is possible when this API gets invoked from the update_kv_cache API where the current_kv only refers to the selected draft and valid token, this will be smaller than the input_ids_slice size.
    """

    warnings.warn("We have deprecated the `pad_to_left` argument and renamed to `trim_from_left` to better suit the functionality, please update your API call if you are passing pad_to_left")
    input_length = input.shape[1]
    # limit the value of this to be non-negative, if it is negative, we assign it to be 0, hence no shifting.
    if layer_indices_to_perform_trimming is None:
        current_pad_length = max(0, (current_key_values[0][1].shape[2] - input_length))
    else:
        current_pad_length = max(0, (current_key_values[layer_indices_to_perform_trimming[0]][1].shape[2] - input_length))
    trimmed_kv = []
    for layer_idx, cur_cache in enumerate(current_key_values):
        # Additional code compare to old: support conv cache
        if len(cur_cache) == 1:
            trimmed_kv.append(cur_cache)
            continue

        (current_key, current_value) = cur_cache
        if layer_indices_to_perform_trimming is None or layer_idx in layer_indices_to_perform_trimming:
            trimmed_key = _shift(current_key, key_concat_axis, current_pad_length, trim_from_left)
            trimmed_value = _shift(current_value, value_concat_axis, current_pad_length, trim_from_left)
            trimmed_kv.append((trimmed_key, trimmed_value))
        elif layer_idx not in layer_indices_to_perform_trimming:
            trimmed_kv.append((current_key, current_value))
    return tuple(trimmed_kv)
