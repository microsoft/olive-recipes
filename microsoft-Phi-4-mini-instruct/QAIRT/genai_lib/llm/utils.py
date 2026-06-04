#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides helper functions for LLM Lib """

from typing import List, Tuple, Union

import torch


def _concat(a, b, dim):
    if isinstance(a, tuple):
        assert len(a) == len(b), 'Unexpected key/value pair'
        return tuple(_concat(ai, bi, dim) for ai, bi in zip(a, b))
    if a is None:
        return b
    if b is None:
        return a
    return torch.cat((a, b), dim=dim)


def _do_concat(a, b, key_dim, value_dim):
    return tuple((_concat(ak, bk, key_dim), _concat(av, bv, value_dim)) for (ak, av), (bk, bv) in zip(a, b))


def _shift(a, dim, shift_size, shift_to_left=True):
    if isinstance(a, tuple):
        return tuple(_shift(ai, dim) for ai in a)
    assert dim in (2, 3), 'Unexpected shift axis'
    if shift_to_left:
        return a[:, :, shift_size:, :] if dim == 2 else a[:, :, :, shift_size:]
    else:
        #this fails when the shift size is 0, in that case, it returns empty tensor, which is opposite of what we want
        #return a[:, :, :-shift_size, :] if dim == 2 else a[:, :, :, :-shift_size]
        orig_len = a.shape[-2] if dim == 2 else a.shape[-1]
        keep_size = orig_len - shift_size
        return a[:, :, :keep_size, :] if dim == 2 else a[:, :, :, :keep_size]


def _do_shift(a, key_dim, value_dim, shift_size):
    return tuple((_shift(k, key_dim, shift_size), _shift(v, value_dim, shift_size)) for k, v in a)

def _get_past_key_value_names(sfx, n_layers, separate_tuple_input_output):
    if not separate_tuple_input_output:
        return ["past_key_values"]
    all = []
    for i in range(n_layers):
        all.append(f'past_key_{i}_{sfx}')
        all.append(f'past_value_{i}_{sfx}')
    return all


def _get_position_emb_names(use_position_embedding_input=True, separate_tuple_input_output=False):
    if separate_tuple_input_output:
        if use_position_embedding_input:
            return ['position_ids_cos', 'position_ids_sin']
    return ['position_ids']


def llm_model_input_output_names(
    num_hidden_layers: int,
    use_position_embedding_input: bool = True,
    separate_tuple_input_output: Union[bool, Tuple[bool, bool]] = False,
    use_input_embedding: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    This function is responsible for returning a list of the model input and output names based on the number of hidden layers for a LLama like signature of inputs

    :param num_hidden_layers: number of hidden layers of the model
    :param use_position_embedding_input: are the position ids supplied to the model in embeddings form (assume sin and cos embedding, if yes)
    :param separate_tuple_input_output: are the inputs passed into the model in tupled format or not
        If passed value is a bool, the value is applied to both input and output
        If passed value tuple of size 2, each bool value is applied separately to input and output
    :param use_input_embedding: do we pass input ids or input embeddings
    :return: model input names and model output names
    """
    if isinstance(separate_tuple_input_output, bool):
        separate_tuple_input, separate_tuple_output = separate_tuple_input_output, separate_tuple_input_output
    elif (
        isinstance(separate_tuple_input_output, tuple)
        and len(separate_tuple_input_output) == 2
    ):
        separate_tuple_input, separate_tuple_output = separate_tuple_input_output
    else:
        raise ValueError('separate_tuple_input_output must be bool or tuple with size 2')

    input_names=['input_ids', 'attention_mask']
    input_names += _get_position_emb_names(use_position_embedding_input=use_position_embedding_input, separate_tuple_input_output=separate_tuple_input)
    output_names = ['logits']
    input_names += _get_past_key_value_names("in", num_hidden_layers, separate_tuple_input)
    output_names += _get_past_key_value_names("out", num_hidden_layers, separate_tuple_output)
    if use_input_embedding:
        input_names += ['inputs_embeds']
        input_names.pop(0)
    return input_names, output_names


def llm_exporter_input_output_names(num_hidden_layers, use_position_embedding_input=True , separate_tuple_input_output=True, use_input_embedding = False):
    '''
    This function returns the input output names passed as input during model export. For downstream use cases, the encodings names are preferred in the untupled format
    params:
    num_hidden_layers: number of hidden layers of the model
    use_position_embedding_input: are the position ids supplied to the model in embeddings form (assume sin and cos embedding, if yes)
    separate_tuple_input_output: are the inputs passed into the model in tupled format or not
    use_input_embedding: do we pass input ids or input embeddings
    '''
    input_names, output_names = llm_model_input_output_names(num_hidden_layers=num_hidden_layers, use_position_embedding_input=use_position_embedding_input , separate_tuple_input_output=separate_tuple_input_output, use_input_embedding = use_input_embedding)
    return input_names, output_names

def llm_search_layers_by_type(model, module_type):
    embedding_layers = []
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            embedding_layers.append(module)
    return embedding_layers

