#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

# =============================================================================
# Cornell-RelaxML Quip-Sharp Hadamard Utility Functions
# Copyright (C) 2023 Cornell RelaxML
#
# This file includes portions of code derived from the Cornell-RelaxML/quip-sharp project:
# https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py
#
# This program is distributed under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================

import torch
from torch import Tensor
from typing import Optional, Tuple
import math


class HadamardTransform(torch.nn.Module):
    """
    Applies the Hadamard transform to the input tensor.
    If `randomized` is True, the Hadamard matrix is randomized by multiplying with a seed vector.
    If `seed` is provided, it is used as the seed vector for the randomized Hadamard matrix.
    If `linear` is True, the Hadamard matrix is applied as a linear transformation
    using a precomputed hadamard as weight matrix.
    Args:
        size (int): Size of the Hadamard matrix. Only powers of two are supported.
        randomized (bool, optional): Whether to use a randomized Hadamard matrix. Defaults to False.
        seed (Tensor, optional): Seed vector for the randomized Hadamard matrix. Defaults to None.
        linear (bool, optional): Whether to apply the Hadamard matrix as a linear transformation. Defaults to False.
        device (torch.device, optional): Device to use for the Hadamard matrix. Defaults to None.
    """

    def __init__(
        self,
        size: int,
        randomized: bool = False,
        seed: Optional[Tensor] = None,
        linear: bool = False,
        device=None,
    ):
        if not _is_pow2(size):
            raise ValueError("size must be a power of 2")
        if randomized and seed is not None:
            raise ValueError("seed must be None if randomized is True")
        super().__init__()
        if seed is None and randomized:
            seed = torch.bernoulli(0.5 * torch.ones(size, device=device)) * 2 - 1
        self.register_buffer("seed", seed)
        self.size = size
        self.scale = 1 / math.sqrt(size)
        if linear:
            self.hadamard = torch.nn.Linear(
                in_features=size,
                out_features=size,
                bias=False,
                device=device,
            )
            self.hadamard.weight.requires_grad = False
            self.hadamard.weight.data.copy_(hadamard_transform(torch.eye(size, device=device)))
        self.eps = 1e-8

    def forward(self, x: Tensor) -> Tensor:
        y = x * self.seed if self.seed is not None else x
        if hasattr(self, "hadamard"):
            # If linear, apply the Hadamard matrix as a linear transformation.
            return self.hadamard(y) * self.scale
        else:
            return hadamard_transform(y, scale=self.scale)

    def apply_inverse(self, x: Tensor, transpose: bool = False) -> Tensor:
        """
        Applies the inverse of the Hadamard transform to the input tensor x.
        For no transpose, equivalent to (X @ H^-1) @ diag(1/s)  = X @ H^T @ diag(1/s), where H is the Hadamard matrix.
        For transpose, equivalent to X @ (H^-1 @ diag(1/s))^T  = X @ diag(1/s) @ H, where H is the Hadamard matrix.

        Args:
            x (Tensor): Input tensor of shape (..., size).
        Returns:
            Tensor: Output tensor of shape (..., size).
        """
        seed = self.seed if self.seed is not None else torch.ones(self.size, device=x.device)
        if transpose:
            # Use H^-1 = H^T, hence fact H^-T = H.
            return hadamard_transform(x / (seed + self.eps), scale=self.scale)
        return hadamard_transform(x, scale=self.scale)[..., : self.size] / (seed + self.eps)


class GroupedHadamardTransform(HadamardTransform):
    """
    Applies the Hadamard transform to a tensor of shape (..., size) in groups of largest power of 2.
    Args:
        size (int): Size of input tensor. This should not be a power of two, but should be decomposable into groups of powers of two.
        randomized (bool, optional): Whether to use a randomized Hadamard matrix. Defaults to False.
        seed (Tensor, optional): Seed vector for the randomized Hadamard matrix. Defaults to None.
        linear (bool, optional): Whether to apply the Hadamard matrix as a linear transformation. Defaults to False.
        device (torch.device, optional): Device to use for the Hadamard matrix. Defaults to None.
    """

    def __init__(
        self,
        size: int,
        randomized: bool = False,
        seed: Optional[Tensor] = None,
        linear: bool = False,
        device=None,
    ):
        if randomized and seed is not None:
            raise ValueError("seed must be None if randomized is True")

        if seed is None and randomized:
            seed = torch.bernoulli(0.5 * torch.ones(size, device=device)) * 2 - 1
        n_groups, groupsize = decompose_for_hadamard_grouped(size)

        super().__init__(size=groupsize, seed=seed, linear=linear, device=device)

        self.n_groups = n_groups
        self.groupsize = groupsize
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        y = x * self.seed if self.seed is not None else x
        if hasattr(self, "hadamard"):
            # If linear, apply the Hadamard matrix as a linear transformation.
            y = y.view(-1, self.n_groups, self.groupsize)
            return (self.hadamard(y) * self.scale).reshape(x.shape)
        else:
            return hadamard_grouped_transform(y, self.n_groups, self.groupsize, scale=self.scale)

    def apply_inverse(self, x: Tensor, transpose: bool = False) -> Tensor:
        """
        Applies the inverse of the Hadamard transform to the input tensor x reshaped to groups.
        For no transpose, equivalent to (X @ H^-1) @ diag(1/s)  = X @ H^T @ diag(1/s), where H is the Hadamard matrix.
        For transpose, equivalent to X @ (H^-1 @ diag(1/s))^T  = X @ diag(1/s) @ H, where H is the Hadamard matrix.

        Args:
            x (Tensor): Input tensor of shape (..., size).
        Returns:
            Tensor: Output tensor of shape (..., size).
        """
        seed = self.seed if self.seed is not None else torch.ones(self.size, device=x.device)
        if transpose:
            # Use H^-1 = H^T, hence fact H^-T = H.
            return hadamard_grouped_transform(
                x / (seed + self.eps),
                self.n_groups,
                self.groupsize,
                scale=self.scale,
            )
        return hadamard_grouped_transform(
            x,
            self.n_groups,
            self.groupsize,
            scale=self.scale,
        ) / (seed + self.eps)


def hadamard_transform(
    X: Tensor,
    scale: float = 1.0,
) -> Tensor:
    """
    Apply Hadamard transform to a tensor and scale it.
    e.g. X @ H.T / sqrt(n), where H is Hadamard matrix and n is the last dimension of X.

    Caution: This function is meant to validate alignment with `fast_hadamard_transform`.
    The einsum-based implementation shows differences up to rtol=1e-2 and is not fully aligned.

    Args:
        X (Tensor): Input tensor.
        scale (float): Scaling factor to apply to the Hadamard transformed tensor.

    Returns:
        Tensor: Hadamard transformed tensor.
    """
    n = X.shape[-1]
    if not _is_pow2(n):
        raise ValueError(
            f"Hadamard transform requires the last dimension to be a power of 2, got {n}."
        )
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > 1:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    return input.view(X.shape) * scale


def hadamard_grouped_transform(x: Tensor, n_groups: int, groupsize: int, scale: float = 1.0) -> Tensor:
    """
    Applied a grouped Hadamard to the last dimension.
    Groups are defined as the largest power of 2 that the dimension decomposes into, e.g. if
    d =  2^n * k, with k non-divisible by 2, then there are k groups of size 2^n.
    Hadamard is applied to each group independently (i.e. the Hadamard matrix is block diagonal)

    To get the sizes, call `decompose_for_hadamard_grouped(x.shape[-1])`
    """

    y = x.reshape(*x.shape[:-1], n_groups, groupsize)
    return hadamard_transform(y, scale=scale).reshape(x.shape)


def _is_pow2(n: int) -> bool:
    """
    Check if a number is a power of 2.

    Args:
        n (int): Number to check.

    Returns:
        bool: True if n is a power of 2, False otherwise.
    """
    return (n & (n - 1) == 0) and (n > 0)


def decompose_for_hadamard_grouped(size: int) -> Tuple[int, int]:
    """
    Decomposes the size into groups of largest power of 2.
    Returns the number of groups and the size of each group.
    e.g. 12 -> 3 groups of size 4
    Args:
        size (int): Size to decompose.
    Returns:
        Tuple[int, int]: Number of groups and size of each group.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    # Find the largest power of two that divides size
    groupsize = 1 << (size.bit_length() - 1)
    while size % groupsize != 0:
        groupsize >>= 1
    n_groups = size // groupsize
    return n_groups, groupsize