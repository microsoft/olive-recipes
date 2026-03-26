
#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Implementation of concepts from the SpinQuant paper: https://arxiv.org/pdf/2405.16406"""

import torch
from torch import nn
import scipy
import math

class R3Hadamard(nn.Linear):
    def __init__(self, head_dim: int):
        super(R3Hadamard, self).__init__(in_features=head_dim, out_features=head_dim, bias=False, dtype=torch.float)
        self.head_dim = head_dim
        self.is_initialized = False

    def initialize_r3_hadamard(self):
        r3_weight = torch.tensor(scipy.linalg.hadamard(self.head_dim) / math.sqrt(self.head_dim), dtype=torch.float)
        if isinstance(self, nn.Linear):
            self.weight.data.copy_(r3_weight.T)
        elif isinstance(self, nn.Conv2d): # For support to ConvInplaceLinear
            self.weight.data.copy_(r3_weight.T[:,:,None,None])
        else:
            raise TypeError(f"Class {self.__class__.__name__} became an instance of {type(self)}, \
                            but only nn.Linear and nn.Conv2d are supported types")
        self.is_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            raise AssertionError(f"{self.__class__.__name__} has not been fully initialized. \
                                 Invoke class method `initialize_r3_hadamard()` before doing a forward pass with this object")
        return super.forward(x)
