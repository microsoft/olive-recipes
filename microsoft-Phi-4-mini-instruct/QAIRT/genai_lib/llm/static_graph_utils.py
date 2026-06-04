#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides utilities for the pipeline to work with static shape requirements for inputs that go into the model """

import torch
from genai_lib.llm.utils import _shift, _concat

def llm_slice_inputs_for_inference(max_input_tokens, model_context_len, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None, past_seen_tokens=None, hidden_states=None, remainder_first=True):
    """
    This function is responsible for slicing the inputs based on the AR and yield them to the user.
    params:
    1. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    2. model_context_len: maximum number of tokens that the model can consume in total
    3. input_ids: input ids sent to the model
    4. inputs_embeds: input embeds sent to the model
    5. attention_mask: attention mask sent to the model
    6. position_ids: position ids sent to the model
    7. hidden_states: hidden states sent to the model
    8. remainder_first: boolean flag which indicates whether we are slicing such that the remainder is in beginning or in end. for an input of size 10 and ARN=3, If true, the remainder is in the beginning [1, 3, 3, 3] else it is in the end [3, 3, 3, 1]

    Note: To be able to ingest all the model_context_len tokens, we need to slice using the left padding and chunk the input into chunks of max_input_tokens
    """

    input_count = 0
    for input in (input_ids, inputs_embeds):
        if input is not None:
            input_count = input_count + 1

    assert input_count == 1, "Should pass either input ids or input embeddings, not both"

    if input_ids is not None:
        input_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        device = input_ids.device
    else:
        input_length = inputs_embeds.shape[1]
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, input_length), dtype = torch.long, device = device)

    if position_ids is None:
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        if past_seen_tokens is not None:
            position_ids += past_seen_tokens

    # If suppose we have an input chunk of size 10, and max_input_tokens (ARN) is 3, then we can either do [1, 3, 3, 3] (remainder_first) or we can do [3, 3, 3, 1] (remainder_last)
    if remainder_first:
        """
        As an example consider:
        ctx_len: 10
        input_len: 10
        max_input_tokens: 3
        KV$ that can be sent into the model is ctx_len-max_input_tokens = 10-3 = 7
        Chunks we will send [1, 3, 3, 3]
        After 1st iteration: accumulated KV$ = 1
        After 2nd iteration: accumulated KV$ = 4
        After 3rd iteration: accumulated KV$ = 7
        
        Now, when sending the last slice of 3, we will either pad it left or right, irrespective of that, the past KV$
        that can flow into the model/ or the KV$ that the current input slice will attend to can only be ctx_len-ARN, hence
        we will only look at 7 which is accumulated accurately until this point.
        
        Hence, we can pass ctx_len worth of input chunk into the model without needing any eviction logic here.
        This is the default behavior.
        """
        for idx in range(0, input_length, max_input_tokens)[::-1]:
            idx = input_length - idx
            slice_beginning = max(0, idx-max_input_tokens)
            output_slice = {
                'attn_mask_slice':attention_mask[:, slice_beginning:idx],
                'position_ids_slice': position_ids[:, slice_beginning:idx],
            }

            if input_ids is not None:
                output_slice['input_ids_slice'] = input_ids[:, slice_beginning:idx]
            else:
                output_slice['inputs_embeds_slice'] = inputs_embeds[:, slice_beginning:idx, :]

            if hidden_states is not None:
                output_slice['hidden_states_slice'] = hidden_states[:, slice_beginning:idx]

            yield output_slice
    else:
        """
        This is the default behavior for Qualla/ on-target
        As an example consider:
        ctx_len: 10
        input_len: 10
        max_input_tokens: 3
        KV$ that can be sent into the model is ctx_len-max_input_tokens = 10-3 = 7
        Chunks we will send [3, 3, 3, 1]
        After 1st iteration: accumulated KV$ = 3
        After 2nd iteration: accumulated KV$ = 6
        After 3rd iteration: accumulated KV$ = 9
        
        Now, when sending the last slice of 1, we will either pad it left or right, irrespective of that, the past KV$
        that can flow into the model/ or the KV$ that the current input slice will attent to can only be ctx_len-ARN, hence
        we will only look at 7 (instead of 9 KV$) and loose information as we need to evict 2 KV$
        
        More importantly, we will have to evict this extra KV$ otherwise we will run into issues.
        """
        for idx in range(0, input_length, max_input_tokens):
            slice_end = min(idx+max_input_tokens, input_length)
            output_slice = {
                'attn_mask_slice': attention_mask[:, idx: slice_end],
                'position_ids_slice': position_ids[:, idx: slice_end],
            }

            if input_ids is not None:
                output_slice['input_ids_slice'] = input_ids[:, idx: slice_end]
            else:
                output_slice['inputs_embeds_slice'] = inputs_embeds[:, idx: slice_end, :]

            if hidden_states is not None:
                output_slice['hidden_states_slice'] = hidden_states[:, idx: slice_end]

            yield output_slice


def llm_pad_inputs(max_input_tokens, input_ids_slice=None, inputs_embeds_slice=None, pad_token=0, pad_embeds=None, pad_to_left=True):
    '''
    This function pads the input_ids/ inputs_embeds since slice may return input_ids/ inputs_embeds that is smaller in length
    than what the model accepts (AR len)

    params:
    1. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    2. input_ids_slice: the current input ids slice that is passed into the model in the current invocation
    3. inputs_embeds_slice: the current input embeds slice that is passed into the model in the current invocation
    4. pad_token: padding token, this is defaulted to 0 to avoid impacting the range of values in the activation tensor
    5. pad_embeds: Tensor with which we pad the inputs_embeds_slice. This is optional and will be used if provided.
        If this is not provided, and we are working with input embeddings, the inputs_embeds_slice tensor will be padded
        with zero and not the pad_token. The reason for this is that the pad_token could be a large non-zero value which
        will impact the range of values in the padded tensor.
    6. pad_to_left: boolean value indicating whether padding is done towards the left or right.

    '''
    input = input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    device = input.device
    input_length = input.shape[1]
    batch_size = input.shape[0]
    shape = (batch_size, max_input_tokens - input_length)

    if pad_embeds is None:
        if inputs_embeds_slice is not None:
            shape += (input.shape[-1],)
            pad_token = 0

        input_extensions = torch.full(
            shape,
            fill_value=pad_token,
            dtype=input.dtype,
            device=device
        )
    else:
        assert input.shape[-1] == pad_embeds.shape[-1]
        # we only want to extract the embeddings dimension from the passed pad_embeddings
        pad_embeds = pad_embeds[-1]
        input_extensions = pad_embeds.view(1, 1, -1).repeat(batch_size, max_input_tokens - input_length, 1).to(dtype=input.dtype, device=device)

    # left padding
    if pad_to_left:
        input = torch.cat((input_extensions, input), dim=1)
    # right padding
    else:
        input = torch.cat((input, input_extensions), dim=1)

    return input

def llm_pad_hidden_states(max_input_tokens: int, hidden_states_slice: torch.Tensor, pad_token=0, pad_to_left=True):
    '''
    This function pads the hidden states since slice may return hidden states that is smaller in length
    than what the model accepts (AR len)

    params:
    1. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    2. hidden_states_slice: the current hidden state slice that is passed into the model in the current invocation
    3. pad_token: padding token, this is defaulted to 0 to avoid impacting the range of values in the activation tensor
    4. pad_to_left: boolean value indicating whether padding is done towards the left or right.

    '''
    pad_shape = list(hidden_states_slice.shape)
    pad_shape[1] = max_input_tokens - hidden_states_slice.shape[1]
    pad = torch.full(
        pad_shape,
        fill_value=pad_token,
        dtype=hidden_states_slice.dtype,
        device=hidden_states_slice.device
    )

    # left padding
    if pad_to_left:
        padded_hidden_states_slice = torch.cat((pad, hidden_states_slice), dim=1)
    # right padding
    else:
        padded_hidden_states_slice = torch.cat((hidden_states_slice, pad), dim=1)

    return padded_hidden_states_slice

def llm_pad_input_attn_mask(attn_mask_slice, max_input_tokens, pad_to_left=True):
    """
    This function pads the 1d attention mask to make it of shape (batch_size, max_input_tokens),

    A: padded current input (0s)
    B: current valid input (1s)

    If the pad_to_left argument is set to True, it means we perform left_padding & produce the attention mask as A|B
    else, pad_to_left=False means we do right padding, & produce the attention mask as B|A

    params:
    attn_mask_slice: the attention mask which corresponds to the current slice of inputs
    max_input_tokens: the maximum tokens that can be sent to the model, in our context represents the AR length
    pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """
    batch_size = attn_mask_slice.shape[0]
    input_padding_length = max_input_tokens - attn_mask_slice.shape[1]

    padded_input_attn_mask = torch.zeros((batch_size, input_padding_length), dtype=torch.long,
                                         device=attn_mask_slice.device)

    #left padding
    if pad_to_left:
        attention_mask = torch.cat((
            padded_input_attn_mask,
            attn_mask_slice
        ),
            dim=1
        )
    # right padding
    else:
        attention_mask = torch.cat((
            attn_mask_slice,
            padded_input_attn_mask
        ),
            dim=1
        )

    return attention_mask


def llm_create_kv_attn_mask(unpadded_past_kv, model_context_len, max_input_tokens, batch_size, device, pad_to_left=True, global_layer_idx = 0):
    """
    This function prepares the 1d attention mask based on the useful past key values seen so far.
    This can be visualized into two sections.
    A | B
    A: padded past kv length (0s)
    B: useful past kv length (1s)

    params:
    unpadded_past_kv: this is the useful accumulated past kv
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    model_context_len: maximum number of tokens that the model can consume in total
    batch_size: batch size of current input
    device: device to place the attention mask
    pad_to_left: boolean value indicating whether padding is done towards the left or right.
    global_layer_idx: an integer representing the index of the global layer (for models like Gauss, where we have sliding window attention, the first layer could be the sliding layer whose shape may not reflect the correct context_len)
    """

    useful_past_kv_length = unpadded_past_kv[global_layer_idx][1].shape[-2] if unpadded_past_kv else 0
    padded_kv_length = (model_context_len - max_input_tokens) - useful_past_kv_length

    useful_past_kv_attn_mask = torch.ones((batch_size, useful_past_kv_length), dtype=torch.long,
                                          device=device)
    padded_kv_attn_mask = torch.zeros((batch_size, padded_kv_length), dtype=torch.long, device=device)

    # left padding
    if pad_to_left:
        attention_mask = torch.cat((
            padded_kv_attn_mask,
            useful_past_kv_attn_mask
        ),
            dim=1
        )
    #right padding
    else:
        attention_mask = torch.cat((
            useful_past_kv_attn_mask,
            padded_kv_attn_mask
        ),
            dim=1
        )
    return attention_mask

def llm_create_1d_attn_mask(attn_mask_past_kv, attn_mask_input, cache_index=None):
    '''
    This function concatenates the attention mask corresponding to the input ids and the past kv together
    params:
    attn_mask_past_kv: the attention mask corresponding to the past kv
    attn_mask_input: the attention mask corresponding to the input (max_input tokens that the model takes)
    cache_index: cache_index determines where should the attn_mask_input be placed. If None, the input_attention mask
    is placed towards the end (assuming concat in the kv update within attention) else it is placed right after the valid kv mask.
    '''
    if cache_index is None:
        attention_mask = torch.cat((attn_mask_past_kv,
                                        attn_mask_input
                                        ),
                                        dim=1
                                        )
    else:
        attention_mask_post_valid_kv = attn_mask_past_kv[:, cache_index:]
        attention_mask_valid_kv = attn_mask_past_kv[:, :cache_index]
        attention_mask = torch.cat((
            attention_mask_valid_kv,
            attn_mask_input,
            attention_mask_post_valid_kv
        ),
            dim=1)


    return attention_mask

def llm_pad_past_kv(dummy_past_kv, unpadded_past_kv, num_hidden_layers, key_concat_axis, value_concat_axis=2, pad_to_left=True):
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

    # if the dummy kv is a list, then we will iterate over that and pad the KV for that layer accordingly. This gives us the flexibility to pad different layers according to it's own ctx_len.
    if isinstance(dummy_past_kv, list):
        assert len(dummy_past_kv) == num_hidden_layers, "Please make sure you pass the dummy KV for each layer"
        padded_key_values = tuple()
        for i in range(num_hidden_layers):
            dummy_past_kv_i = dummy_past_kv[i]
            # get the useful past kv length of the ith layer
            useful_past_kv_length = unpadded_past_kv[i][1].shape[-2] if unpadded_past_kv else 0

            # trim the dummy kv corresponding to that particular layer based on it's useful kv length
            # trimmed dummy kv is the final length dummy kv that will be concatenated to the unpadded_past_kv either to the left or to the right.
            trimmed_dummy_kv = (_shift(dummy_past_kv_i[0], key_concat_axis, useful_past_kv_length),
                                _shift(dummy_past_kv_i[1], value_concat_axis, useful_past_kv_length))
            if unpadded_past_kv:
                if pad_to_left:
                    padded_key_values_i = (_concat(trimmed_dummy_kv[0], unpadded_past_kv[i][0], key_concat_axis),
                                               _concat(trimmed_dummy_kv[1], unpadded_past_kv[i][1], value_concat_axis))
                else:
                    padded_key_values_i = (_concat(unpadded_past_kv[i][0], trimmed_dummy_kv[0], key_concat_axis),
                                               _concat(unpadded_past_kv[i][1], trimmed_dummy_kv[1], value_concat_axis))

                padded_key_values += (padded_key_values_i, )
            else:
                padded_key_values += (trimmed_dummy_kv, )
        return  padded_key_values

    else:
        useful_past_kv_length = unpadded_past_kv[0][1].shape[-2] if unpadded_past_kv else 0

        # trimmed dummy kv is the final length dummy kv that will be concatenated to the unpadded_past_kv either to the left or to the right.
        trimmed_dummy_kv = (_shift(dummy_past_kv[0], key_concat_axis, useful_past_kv_length), _shift(dummy_past_kv[1], value_concat_axis, useful_past_kv_length))
        if unpadded_past_kv:
            if pad_to_left:
                padded_key_values = tuple((_concat(trimmed_dummy_kv[0], unpadded_past_kv[i][0], key_concat_axis),
                                           _concat(trimmed_dummy_kv[1], unpadded_past_kv[i][1], value_concat_axis)) for i in range(num_hidden_layers))
            else:
                padded_key_values = tuple((_concat(unpadded_past_kv[i][0], trimmed_dummy_kv[0], key_concat_axis),
                                           _concat(unpadded_past_kv[i][1], trimmed_dummy_kv[1], value_concat_axis)) for i in
                                          range(num_hidden_layers))
            return padded_key_values
        return tuple(trimmed_dummy_kv for _ in range(num_hidden_layers))

def llm_get_dummy_kv(batch_size,num_key_value_heads, head_dim, key_concat_axis, device, dtype=torch.float32, cache_len = None, model_context_len=None, max_input_tokens=None ):
    """
    This function determines the shape of the dummy kv using the required arguments which reflect model config
    Returns the dummy kv of fixed size each time (for a single layer). This will be used for padding the passed past kv

    params:
    batch_size: the batch size needed to create dummy kv
    model_context_len : model_context_len: maximum number of tokens that the model can consume in total
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    num_key_value_heads: the number of key value heads
    head_dim: dimension at each head
    key_concat_axis: the axis to which we want to append the keys
    device: the device to place dummy kv on, this is inferred from the unpadded_past_kv tensor if it is not None
    """

    def _cache(shape):
        return torch.zeros(shape, device=device, dtype=dtype)

    if cache_len is None:
        cache_len = model_context_len-max_input_tokens

    value = (batch_size, num_key_value_heads,cache_len , head_dim)
    key = (value[0], value[1], value[3], value[2]) if key_concat_axis == 3 else tuple(value)
    return (_cache(key), _cache(value))

def _llm_trim_padded_tensor(tensor, input_length, pad_axis=1, pad_to_left=True):
    """
    This function is responsible for stripping the non-useful values from the returned tensor (e.g., logits or hidden states)
    since our prepared model returns fixed length tensor
    params:
        tensor: current tensor returned from the model (e.g., logits or hidden states)
        input_length: length of the valid portion of tensor
        pad_axis: dimension index of padding
        pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """
    # left padding so we remove the logits from the left & return the valid input length from the end
    if pad_to_left:
        trimmed_tensor = torch.narrow(tensor, pad_axis, tensor.shape[pad_axis] - input_length, input_length)
    # right padding, so we extract the valid input_length from the beginning
    else:
        trimmed_tensor = torch.narrow(tensor, pad_axis, 0, input_length)
    return trimmed_tensor

def llm_trim_pad_logits(cur_logits, input_ids_slice=None, inputs_embeds_slice=None, pad_to_left=True):
    """
    This function is responsible for stripping the non-useful logits from the returned logits since our prepared model returns fixed length logits
    params:
        cur_logits: current logits returned from the model
        input_ids_slice: current input ids slice which is not padded
        pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """
    input = input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    input_length = input.shape[1]

    return _llm_trim_padded_tensor(cur_logits, input_length=input_length, pad_to_left=pad_to_left)

def llm_trim_padded_hidden_states(hidden_states, input_ids_slice=None, inputs_embeds_slice=None, pad_to_left=True):
    """
    This function is responsible for stripping the non-useful hidden states from the returned hidden states
    since our prepared model returns fixed length hidden states
    params:
        hidden_states: current hidden states returned from the model
        input_ids_slice: current input ids slice which is not padded
        pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """
    input = input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    input_length = input.shape[1]

    return _llm_trim_padded_tensor(hidden_states, input_length=input_length, pad_to_left=pad_to_left)

def llm_get_position_ids_from_attention_mask(attention_mask, max_input_tokens, model_context_len, cache_index=None):
    """
    This function computes the position ids for the tokens being fed into the model from the 1d_attn_mask.

    params:
    attention_mask: takes in the prepared attention mask needed to deduce the position ids
    max_input_tokens: the maximum tokens that can be sent to the model, in our context represents the AR length
    model_context_len : the maximum context length that can be sent to the model (this is the HF maximum length of the context)
    cache_index: the index for the starting position of kvcaches
    """

    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    position_ids = position_ids.clip(0, model_context_len - 1)
    if cache_index is None:
        position_ids = position_ids[..., -max_input_tokens:]
    else:
        position_ids = position_ids[..., cache_index:cache_index+max_input_tokens]
    return position_ids

def llm_pad_position_ids(position_ids_slice, max_input_tokens, pad_value=0, pad_to_left = True):
    """
    This function pads the position_ids since slice may return position_ids that is smaller than what the model accepts (AR len)

    params:
    position_ids_slice: the current position_ids slice that is passed into the model in the current invocation
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    pad_value: padding value, this is defaulted to 0
    pad_to_left: boolean value indicating whether padding is done towards the left or right.

    """

    assert position_ids_slice is not None
    assert position_ids_slice.dim() == 2

    batch_size, pos_ids_len = position_ids_slice.shape

    if pos_ids_len < max_input_tokens:
        pad_pos_ids = torch.full((batch_size, max_input_tokens-pos_ids_len), pad_value,
                                 dtype=position_ids_slice.dtype, device=position_ids_slice.device)

        if pad_to_left:
            position_ids = torch.cat((pad_pos_ids, position_ids_slice), dim=-1)
        else:
            position_ids = torch.cat((position_ids_slice, pad_pos_ids), dim=-1)

        return position_ids
    else:
        return position_ids_slice


def slice_tensors(slice_length, max_length, tensor_dict, remainder_first=True, **kwargs):
    """
    Slices tensors in a dictionary along specified dimensions into smaller chunks.

    Parameters:
    -----------
    slice_length : int
        The length of each slice.
    max_length : int
        The total length to be sliced from each tensor.
    tensor_dict : dict
        A dictionary where keys are variable names and values are tuples of the form (tensor, slice_dim),
        where `tensor` is a PyTorch tensor and `slice_dim` is the dimension along which to slice.
    remainder_first : bool, optional (default=True)
        If True, the remainder (if any) is included in the first slice. Otherwise, it's included in the last slice.
    **kwargs : dict
        Additional keyword arguments (not used in this function but included for extensibility).

    Yields:
    -------
    dict
        A dictionary of sliced tensors corresponding to each slice.
    """
    remainder = max_length % slice_length
    num_full_slices = max_length // slice_length
    num_slices = num_full_slices + (1 if remainder > 0 else 0)

    for i in range(num_slices):
        sliced_dict = {}

        # Determine start and end indices for the current slice
        if remainder_first:
            if i == 0 and remainder > 0:
                start_idx = 0
                end_idx = remainder
            else:
                start_idx = remainder + (i - 1) * slice_length if remainder > 0 else i * slice_length
                end_idx = start_idx + slice_length
        else:
            if i < num_full_slices:
                start_idx = i * slice_length
                end_idx = start_idx + slice_length
            else:
                start_idx = num_full_slices * slice_length
                end_idx = max_length

        # Slice each tensor in the dictionary
        for var_name, (tensor, slice_dim) in tensor_dict.items():
            assert isinstance(tensor, torch.Tensor), \
                f"Input {var_name} is not a tensor, but a {type(tensor)}"

            # Adjust end index if it exceeds tensor size
            if end_idx > tensor.size(slice_dim):
                end_idx = tensor.size(slice_dim)

            # Skip if the slice would be empty or invalid
            if end_idx - start_idx <= 0:
                return

            # Perform the slicing using torch.narrow
            sliced_dict[var_name] = tensor.narrow(slice_dim, start_idx, end_idx - start_idx)

        yield sliced_dict
