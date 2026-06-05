# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""This file contains utilities for manipulating quantsim quantizers"""

from typing import Sequence, Set, Optional, Union

import torch
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import QuantizedConv2d, QuantizedLinear
from aimet_torch.v2.nn.modules.custom import QuantizedMultiply, QuantizedAdd


def freeze_quantizers(
        sim: QuantizationSimModel,
        target_types: Union[Sequence[str], str] = ('input', 'output', 'param'),
        return_act_encodings: bool = False,
        return_param_encodings: bool = False,
        target_layers: Optional[Union[Set[str], Sequence[str]]] = None,
):
    """
    Freezes quantizer encodings and optionally return the frozen encodings
    :param sim: The sim object of which the quantizers are set to be non-overwritable
    :param target_types: The type of quantizers to freeze. Can be any of 'input', 'output', 'param' or a sequence of
    any combinations of these options
    :param return_act_encodings: Whether to return the activation encodings
    :param return_param_encodings: Whether to return the parameter encodings
    :param target_layers [Optional]: The layers to freeze. If not specified, default to freeze all layers
    :return: Optionally return the frozen encodings
    """
    if isinstance(target_types, str):
        target_types = [target_types]

    if target_layers is not None and not isinstance(target_layers, set):
        target_layers = set(target_layers)

    for layer_name, qmodule in sim.named_qmodules():
        # Skip layers if not found in frozen_layers
        if target_layers is not None and layer_name not in target_layers:
            continue

        for target_type in target_types:
            if target_type.lower() == 'input':
                for input_quantizer in qmodule.input_quantizers:
                    if input_quantizer is not None:
                        input_quantizer.allow_overwrite(False)

            if target_type.lower() == 'output':
                for output_quantizer in qmodule.output_quantizers:
                    if output_quantizer is not None:
                        output_quantizer.allow_overwrite(False)

            if target_type.lower() == 'param':
                for param_quantizer in qmodule.param_quantizers.values():
                    if param_quantizer is not None:
                        param_quantizer.allow_overwrite(False)

    if return_act_encodings or return_param_encodings:
        act_encodings, param_encodings = sim.get_activation_param_encodings()

        return_encodings = {}

        if return_act_encodings:
            return_encodings['activation_encodings'] = act_encodings

        if return_param_encodings:
            return_encodings['param_encodings'] = param_encodings

        return return_encodings

def set_lora_quantizer(
        sim: QuantizationSimModel,
        target_modules: Union[Set[str], Sequence[str]],
        param_bitwidth: int = 8,
        scaling_min: float = 0.0,
        scaling_max: float = 1.0,
):
    """
    Set the quantizer range for LoRA scalings and bitwidth of LoRA parameters
    :param sim: The sim object of which the LoRA quantizers are set to the provided range
    :param target_modules: The LoRA target_modules. We use target_modules to do substring
    match for the sim.model's layer names in order to find the correspoding lora layers
    :param param_bitwidth: The bitwidth of the LoRA param quantizers
    :param scaling_min: The min value to set for LoRA scalings quantizers
    :param scaling_max: The max value to set for LoRA scalings quantizers
    :return: None. Quantizers are set in-place
    """
    for layer_name, module in sim.named_qmodules():
        if (isinstance(module, (QuantizedConv2d, QuantizedLinear, QuantizedMultiply)) and
                any(map(lambda target_module: target_module in layer_name, target_modules))):
            if 'lora' in layer_name and isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                # Set bitwidth for LoRA A/B (aka up/down) modules
                module.param_quantizers['weight'].bitwidth = param_bitwidth
            if isinstance(module, QuantizedMultiply):
                # Set range of the LoRA multiplication layer
                module.input_quantizers[1].set_range(
                    torch.as_tensor(scaling_min),
                    torch.as_tensor(scaling_max)
                )
                # Freeze Quantizer
                module.input_quantizers[1].allow_overwrite(False)

def propagate_lora_add(sim: QuantizationSimModel, target_modules: Union[Set[str], Sequence[str]]):
    """
    Propagate LoRA add's output quantizers from its base layer via unifying their output_quantizers
    :param sim: The sim object of which the LoRA add quantizers are to be unified
    :param target_modules: The LoRA target_modules. We use target_modules to do substring matching.
    The substrings in target_modules must uniquely identify each layer s.t. we can find the unique
    match of the layer in sim
    :return: None. Propagation happens in-place
    """
    base_output_quantizers = {}
    lora_add_modules = {}

    for layer_name, module in sim.named_qmodules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear, QuantizedAdd)):
            for target_module in target_modules:
                if target_module in layer_name:
                    if 'base_layer' in layer_name and isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                        base_output_quantizers[target_module] = module.output_quantizers
                    if isinstance(module, QuantizedAdd):
                        lora_add_modules[target_module] = module
                    break

    for target_module, base_output_quantizer in base_output_quantizers.items():
        lora_add_modules[target_module].output_quantizers = base_output_quantizer
