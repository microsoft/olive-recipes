#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" common utilities and class implementation for generating test vector, accuracy analysis """

import sys

import numpy as np
import torch

def get_sqnr_nptensor(
    fp_tensor: np.ndarray, sim_tensor: np.ndarray, eps: float, sample_wise: bool = False
) -> float:
    """
    Computes SQNR

    :param fp_tensor: FP32 np array
    :param sim_tensor: np array with QDQ noise.
    :param eps: the smallest positive value to avoid div-by-zero error.
    :param sample_wise: Specifies whether to calculate SQNR per sample before averaging.
        If set to True, It mitigates the dominance of stronger signal samples in the SQNR calculation.
        If set to False, SQNR is calculated across all elements. Defaults to False for backward compatibility.
    :return: Sigal-to-quantization noise ratio in dB scale.
    """
    if fp_tensor.shape != sim_tensor.shape:
        raise ValueError('Both tensors must have the same shape')

    if sample_wise and fp_tensor.ndim < 2:
        raise ValueError(
            'For sample-wise calculation, both tensors must have at least two dimensions'
        )

    sim_error = fp_tensor - sim_tensor

    if sample_wise:
        # Sample-wise calculation (dims_except_sample_dim)
        axis = tuple(range(fp_tensor.ndim)[1:])
    else:
        axis = None

    exp_noise = (sim_error**2).mean(axis=axis) + eps
    exp_signal = (fp_tensor**2).mean(axis=axis)
    sqnr_db = 10 * (np.log10(exp_signal) - np.log10(exp_noise))
    return sqnr_db.mean()


def get_sqnr(
    fp_tensor: torch.Tensor,
    sim_tensor: torch.Tensor,
    eps: float = sys.float_info.min,
    sample_wise: bool = False,
) -> float:
    """
    Computes SQNR

    :param fp_tensor: FP32 torch tensor
    :param sim_tensor: torch tensor with QDQ noise.
    :param eps: the smallest positive value to avoid div-by-zero error.
    :param sample_wise: Specifies whether to calculate SQNR per sample before averaging.
        If set to True, it mitigates the dominance of stronger signal samples in the SQNR calculation.
        If set to False, SQNR is calculated across all elements. Defaults to False for backward compatibility
    :return: Sigal-to-quantization noise ratio in dB scale.
    """
    return get_sqnr_nptensor(
        np_tensor(fp_tensor), np_tensor(sim_tensor), eps, sample_wise
    )


def np_tensor(tensor):
    """
    :return: numpy tensor(s) from torch tensor(s)
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    if isinstance(tensor, (list, tuple)):
        cls = tuple if isinstance(tensor, tuple) else list
        return cls(np_tensor(x) for x in tensor)

    if isinstance(tensor, dict):
        return {key: np_tensor(value) for key, value in tensor.items()}

    return tensor
