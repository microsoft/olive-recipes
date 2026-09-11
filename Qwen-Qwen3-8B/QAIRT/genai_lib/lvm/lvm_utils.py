#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Utilities to manage AIMET LVM pipeline elements """

import torch
from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def adjust_buffer_batch_size(model: torch.nn.Module, batch_size: int, buffer_names: list[str] = '*') -> None:
    """
    Adjusts all buffers in a torch.nn.Module to align with desired inference batch size
    :param model: model to be adjusted inplace.
    :param batch_size: batch size to set buffer to.
    :param buffer_names: list of which buffers to update batch-size, or '*' to update all buffers
    :return: None
    """
    all_buffers = [(name, buffer.ndim) for name, buffer in model.named_buffers() if name in buffer_names or buffer_names == '*']
    for name, ndim in all_buffers:
        shape = torch.ones(ndim, dtype=torch.int64) * -1
        shape[0] = batch_size
        model._buffers[name] = model._buffers[name][:1].expand(tuple(shape))
    logger.info(f'Adjusted buffers {[name for name, ndim in all_buffers]} to have batch size {batch_size}')


def clip_activations(quantizer, threshold, clip_method):
    if quantizer.get_scale() > threshold:
        old_min = quantizer.get_min()
        old_max = quantizer.get_max()
        old_scaling = quantizer.get_scale()

        if clip_method == 'minside':
            k = (old_max - threshold / old_scaling * (old_max - old_min)) / old_min
            new_max = old_max
            new_min = old_min * k
        elif clip_method == 'twoside':
            k = threshold / old_scaling
            new_max = old_max * k
            new_min = old_min * k
        else:
            raise ValueError(f'{clip_method} is not supported! `clip_method` should be a choice of [`minside`, `twoside`]')

        quantizer.set_range(new_min, new_max)
