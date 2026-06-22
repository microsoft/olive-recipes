# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" This file provides common utilities to replace Linear layers with Conv2d layers. """

import warnings
import torch
from genai_lib.common.dev.utils import rsetattr
from transformers.pytorch_utils import Conv1D

class ConvInplaceLinear(torch.nn.Conv2d):
    """ Convolution module that replaces a Linear layer inplace
    We inherit from the torch.nn.Conv2d so that the resulting module can still behave like a leaf module when adding Lora Adapters.
    This will equip us to use the same Lora Config for the adapted model as the original model config.
    """

    def __init__(self, mod):
        if isinstance(mod, torch.nn.Linear):
            weight, bias = mod.weight, mod.bias
        elif isinstance(mod, Conv1D):
            weight, bias = mod.weight.T, mod.bias

        self.out_features, self.in_features = weight.shape

        super(ConvInplaceLinear, self).__init__(
            self.in_features,
            self.out_features,
            1,
            dtype=mod.weight.dtype,
            bias=True if bias is not None else False
        )

        self.weight.data.copy_(weight.data[:, :, None, None])
        if bias is not None:
            self.bias.data.copy_(bias.data)
        self.to(mod.weight.data.device)

    def forward(self, x: torch.Tensor, scale: float = 1.0):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(-1).permute(0, 2, 3, 1)  # (emb_dim, C) -> (1, C, 1, emb_dim)
        elif ndim == 3:
            x = x.unsqueeze(-1).permute(0, 2, 3, 1)  # (B, emb_dim, C) -> (B, C, 1, emb_dim)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
            warnings.warn(f"{self.__class__.__name__} received an unexpected 4d input, assuming channels-last and proceeding.")
        else:
            raise NotImplementedError(f"{self.__class__.__name__} could not handle input with shape {x.shape}")

        x = super().forward(x)

        if ndim == 2:
            return x.permute(0, 3, 1, 2).squeeze(-1).squeeze(0)  # (1, C, 1, emb_dim) -> # (emb_dim, C)
        elif ndim == 3:
            return x.permute(0, 3, 1, 2).squeeze(-1)  # (1, C, 1, emb_dim) -> # (B, emb_dim, C)
        elif ndim == 4:
            x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        return x


def replace_linears_with_convs(model: torch.nn.Module) -> torch.nn.Module:
    """
    Helper function to replace all linear modules with equivalent conv modules
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            conv_layer = ConvInplaceLinear(module)
            rsetattr(model, name, conv_layer)

    return model