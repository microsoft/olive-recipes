#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Implementation for handling LoRA adapters added using PEFT """

import torch

from packaging import version

from peft.tuners.lora.layer import LoraLayer as PeftLoraLayer
from peft.tuners.lora.layer import Conv2d as PeftConv2d
import aimet_torch.utils as aimet_torch_utils
from aimet_torch.peft import LoraLayer as AimetLoraLayer
from aimet_torch.elementwise_ops import Add, Multiply
from genai_lib.common.dev.utils import get_aimet_version

class LoraLayerNoCast(AimetLoraLayer):
    """
    Quantizable lora layer

    This is the same implementation with the AIMET LoRA layers except that it removes
    data type casting in the forward function
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, lora_layer: PeftLoraLayer):
        """
        :param lora_layer: Lora layer we want to replace
        """
        super().__init__(lora_layer)

        self.add_lora_to_res = torch.nn.ModuleList([Add() for _ in range(len(self.lora_A))])
        self.mul_scale = torch.nn.ModuleList([Multiply() for _ in range(len(self.lora_A))])

        # reshape lora scaling to be broadcastable when it's a vector
        for i, scaling in enumerate(self.scaling):
            if isinstance(scaling, torch.Tensor) and scaling.size():
                self.scaling[i] = torch.nn.Parameter(
                    scaling.view(1, scaling.size()[0], 1, 1), requires_grad=False
                ).to(scaling.device)

        # self.scaling = torch.nn.ParameterList(self.scaling) if isinstance(self.scaling, list) else self.scaling

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """ Forward pass for replaced layer"""
        result = self.base_layer(x, *args, **kwargs)
        # torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if not self.active_adapters[active_adapter]:
                continue

            adapter_index = self.adapter_name_to_index[active_adapter]

            lora_A = self.lora_A[adapter_index]
            lora_B = self.lora_B[adapter_index]
            dropout = self.lora_dropout[adapter_index]
            # TODO: remove the to(device) once MPP error fixed
            scaling = self.scaling[adapter_index].to(lora_A.weight.device)
            # x = x.to(lora_A.weight.dtype)

            result = self.add_lora_to_res[adapter_index](
                result, lora_B(self.mul_scale[adapter_index](lora_A(dropout(x)), scaling.detach()))
            )

        # result = result.to(torch_result_dtype)
        return result

class LoraLayer4d(AimetLoraLayer):
    """
    Quantizable lora layer
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, lora_layer: PeftLoraLayer):
        """
        :param lora_layer: Lora layer we want to replace
        """
        super().__init__(lora_layer)

        self.add_lora_to_res = torch.nn.ModuleList([Add() for _ in range(len(self.lora_A))])
        self.mul_scale = torch.nn.ModuleList([Multiply() for _ in range(len(self.lora_A))])

        # reshape lora scaling to be broadcastable when it's a vector
        for i, scaling in enumerate(self.scaling):
            if isinstance(scaling, torch.Tensor) and scaling.size():
                self.scaling[i].data = scaling.data.view(1, scaling.size()[0], 1, 1)
                # self.register_parameter(f'scaling_{i}', self.scaling[i])

        # TODO: Eventually we want to enable the below snippet but MPP errors out for now. See more in MORPH-19557
        # self.scaling = []
        # for i, scaling in enumerate(lora_layer.scaling.values()):
        #     if isinstance(scaling, torch.Tensor) and scaling.size():
        #         self.scaling.append(torch.nn.Parameter(
        #             torch.as_tensor(scaling).view(1, scaling.size()[0], 1, 1), requires_grad=False
        #         ).to(self.base_layer.weight.device))

        # self.scaling = torch.nn.ParameterList(self.scaling) if isinstance(self.scaling, list) else self.scaling

        self._become_4d()

    def _become_4d(self):
        base_layer_conv = torch.nn.Conv2d(self.base_layer.in_channels if hasattr(self.base_layer, 'in_channels') else self.base_layer.in_features,
                                        self.base_layer.out_channels if hasattr(self.base_layer, 'out_channels') else self.base_layer.out_features,
                                            1, bias=True if self.base_layer.bias is not None else False)
        base_weight = self.base_layer.weight.reshape(self.base_layer.weight.shape[:2])
        base_layer_conv.weight.data.copy_(base_weight[:,:,None,None])
        if self.base_layer.bias is not None:
            base_layer_conv.bias.data.copy_(self.base_layer.bias.data)
        base_layer_conv.to(self.base_layer.weight.data.device)
        self.base_layer = base_layer_conv

        for idx in range(len(self.lora_A)):
            lora_A_conv = torch.nn.Conv2d(self.lora_A[idx].in_channels if hasattr(self.lora_A[idx], 'in_channels') else self.lora_A[idx].in_features,
                                            self.lora_A[idx].out_channels if hasattr(self.lora_A[idx], 'out_channels') else self.lora_A[idx].out_features,
                                                1, bias=True if self.lora_A[idx].bias is not None else False)
            lora_B_conv = torch.nn.Conv2d(self.lora_B[idx].in_channels if hasattr(self.lora_B[idx], 'in_channels') else self.lora_B[idx].in_features,
                                            self.lora_B[idx].out_channels if hasattr(self.lora_B[idx], 'out_channels') else self.lora_B[idx].out_features,
                                                1, bias=True if self.lora_B[idx].bias is not None else False)
            lora_A_weight = self.lora_A[idx].weight.reshape(self.lora_A[idx].weight.shape[:2])
            lora_B_weight = self.lora_B[idx].weight.reshape(self.lora_B[idx].weight.shape[:2])
            lora_A_conv.weight.data.copy_(lora_A_weight[:,:,None,None])
            lora_B_conv.weight.data.copy_(lora_B_weight[:,:,None,None])
            if self.lora_A[idx].bias is not None: lora_A_conv.bias.data.copy_(self.lora_A[0].bias.data)
            if self.lora_B[idx].bias is not None: lora_B_conv.bias.data.copy_(self.lora_B[0].bias.data)
            lora_A_conv.to(self.lora_A[idx].weight.data.device)
            lora_B_conv.to(self.lora_B[idx].weight.data.device)
            self.lora_A[idx] = lora_A_conv
            self.lora_B[idx] = lora_B_conv

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """ Forward pass for replaced layer """
        if not x.ndim == 4:
            raise ValueError(f"{self.__class__.__name__} expects 4D inputs formatted (NCHW), received inputs of shape {x.shape}")

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if not self.active_adapters[active_adapter]:
                continue

            adapter_index = self.adapter_name_to_index[active_adapter]

            lora_A = self.lora_A[adapter_index]
            lora_B = self.lora_B[adapter_index]
            dropout = self.lora_dropout[adapter_index]
            # TODO: remove the to(device) once MPP error fixed
            scaling = self.scaling[adapter_index].to(lora_A.weight.device)
            x = x.to(lora_A.weight.dtype)

            result = self.add_lora_to_res[adapter_index](
                result, lora_B(self.mul_scale[adapter_index](lora_A(dropout(x)), scaling.detach()))
            )

        result = result.to(torch_result_dtype)
        return result

def _is_peft_lora_layer(layer: torch.nn.Module) -> bool:
    """
    Check if the layer is instance of peft.tuners.lora.layer.LoraLayer
    :param layer: The torch layer to be checked
    :return: If the layer is instance of peft.tuners.lora.layer.LoraLayer
    """
    return isinstance(layer, PeftLoraLayer)

def _replace_peft_lora_modules(model: torch.nn.Module, replace_layer: torch.nn.Module) -> None:
    """
    Replace PEFT LoRA modules with the target replace_layer
    :param model: The model on which to replace with LoRA modules
    :param replace_layer: The target layer to replace with
    :return: None. Replacement happens in-place
    """

    aimet_version = get_aimet_version()

    if aimet_version >= version.Version('2.1'):
        # For aimet_torch 2.1.x or greater
        aimet_torch_utils.replace_modules(model, _is_peft_lora_layer, replace_layer)
    else:
        # For aimet_torch 2.0.x or earlier
        aimet_torch_utils.replace_modules_of_type1_using_constructor(model, PeftLoraLayer, replace_layer)
        aimet_torch_utils.replace_modules_of_type1_using_constructor(model, PeftConv2d, replace_layer)

def replace_lora_layers_with_4d_quantizable_layers(model: torch.nn.Module):
    """
    Utility to replace lora layers with Quantizable Lora layers

    :param model: PEFT model
    """
    _replace_peft_lora_modules(model, LoraLayer4d)


def replace_lora_layers_with_no_cast_quantizable_layers(model: torch.nn.Module):
    """
    Utility to replace lora layers with Quantizable Lora layers

    :param model: PEFT model
    """
    _replace_peft_lora_modules(model, LoraLayerNoCast)
