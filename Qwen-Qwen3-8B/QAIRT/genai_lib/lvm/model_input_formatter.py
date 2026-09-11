#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" common utilities and class implementation for model inputs adaptation """
from typing import Dict
import torch
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def all_equal(iterable):
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def _recursor_set_batch_to_1(input_dict, batch_idx):
    """
    recursively shrink an input_dict from batch size B to 1,
    selecting an specific entry withing the batch size, based on batch_idx
    :param input_dict: flattened input tensor dictionary
    :param batch_idx: which batch idx to select
    :return: dictionary adjusted for batch=1
    """
    input_batch_sizes = (value.shape[0] for value in input_dict.values() if isinstance(value, torch.Tensor) and value.ndim>1)
    assert all_equal(input_batch_sizes), "All inputs must have same batch size"

    batch_1_input_dict = {}
    for k in list(input_dict.keys()):
        if isinstance(input_dict[k], dict):
            batch_1_input_dict[k] = _recursor_set_batch_to_1(input_dict[k], batch_idx)
        elif isinstance(input_dict[k], torch.Tensor) and input_dict[k].ndim >= 2:
            batch_1_input_dict[k] = torch.unsqueeze(input_dict[k][batch_idx], 0)
        elif not isinstance(input_dict[k], bool) and input_dict[k] is not None:
            batch_1_input_dict[k] = torch.tensor(input_dict[k]).reshape(-1)
    return batch_1_input_dict


def generator_set_batch_to_1(input_dict):
    """
    generator that consumes an input_dict with data having
    batch size B and yields samples having batch size 1
    :param input_dict: flattened input tensor dictionary
    :return: tuple of Tensor with 4-d tensor adjusted for batch=1
    """
    input_batch_sizes = (value.shape[0] for value in input_dict.values() if isinstance(value, torch.Tensor) and value.ndim>1)
    assert all_equal(input_batch_sizes), "All inputs must have same batch size"
    input_batch_sizes = (value.shape[0] for value in input_dict.values() if isinstance(value, torch.Tensor) and value.ndim>1)

    for idx in range(next(input_batch_sizes)):
        batch_1_input_dict = {}
        for k in list(input_dict.keys()):
            if isinstance(input_dict[k], dict):
                batch_1_input_dict[k] = _recursor_set_batch_to_1(input_dict[k], batch_idx=idx)
            elif isinstance(input_dict[k], torch.Tensor) and input_dict[k].ndim >= 2:
                batch_1_input_dict[k] = torch.unsqueeze(input_dict[k][idx], 0)
            elif not isinstance(input_dict[k], bool) and input_dict[k] is not None:
                batch_1_input_dict[k] = torch.tensor(input_dict[k]).reshape(-1)
        yield batch_1_input_dict
