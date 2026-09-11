# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""This file contains utilities for reconstructing MPP LoRA config and weights"""

import copy
import json
from typing import Any
from pathlib import Path

import torch
from peft.tuners.tuners_utils import check_target_module_exists
from peft.tuners.lora.layer import LoraLayer as PeftLoraLayer
from genai_lib.common.dev.peft.peft import replace_lora_layers_with_no_cast_quantizable_layers


def adapt_base_encodings_for_peft_model(
        base_torch_encodings_path: Path | str,
        full_mpp_target_module_names: set,
        verbose: bool = False,
):
    """
    Load base encodings and modify target module names (add '.base_layer' suffix). The returned base encodings
    can be reused in PEFT model created from prepared base model only.
    :param base_torch_encodings_path: Path to base model torch encodings
    :param full_mpp_target_module_names: Set of MPP layer names to which the suffix '.base_layer' should be added
    :param verbose: If True, print additional information during processing
    :return: Modified base encodings
    """
    if not isinstance(base_torch_encodings_path, Path):
        base_torch_encodings_path = Path(base_torch_encodings_path)

    with open(base_torch_encodings_path) as torch_encodings_json_data:
        base_encodings = json.load(torch_encodings_json_data)

    # Replace target module names in base encodings with .base_layer suffix
    if verbose:
        print(f'Adapting activation encodings for {base_torch_encodings_path.name}')
    _add_peft_suffix_to_encodings(base_encodings['activation_encodings'], full_mpp_target_module_names, verbose)

    if verbose:
        print(f'Adapting param encodings for {base_torch_encodings_path.name}')
    _add_peft_suffix_to_encodings(base_encodings['param_encodings'], full_mpp_target_module_names, verbose)

    return base_encodings


def adapt_lora_config_for_prepared_model(lora_config, name_to_module_dict):
    """
    Adapt LoRA config for prepared model. It changes the target modules to reflect the prepared model names
    :param lora_config: LoRA config for non prepared model
    :param name_to_module_dict: name to module dict obtained from model preparation
    return: The newly constructed LoRA config
    """
    new_lora_config = copy.deepcopy(lora_config)
    new_lora_config.target_modules = set()

    for name in name_to_module_dict:
        if check_target_module_exists(lora_config, name):
            prepared_name = _find_prepared_model_module_name(name, name_to_module_dict)
            new_lora_config.target_modules.add(prepared_name)

    return new_lora_config


def get_lora_weights_for_prepared_model(weights_before_mpp, name_to_module_dict, adapter_name):
    """
    Remap the LoRA weights from pre-MPP to post-MPP namespace
    :param weights_before_mpp: The LoRA weights before MPP is applied
    :param name_to_module_dict: The name mapping from the MPP's .json file
    :param adapter_name: The adapter_name of the LoRA adapter
    return: The remapped LoRA weights
    """
    weights_after_mpp = {}

    for key in name_to_module_dict:
        for name in weights_before_mpp:
            if key in name:
                prepared_name = _find_prepared_model_module_name(key, name_to_module_dict)
                if 'lora_A' in name:
                    weights_after_mpp['.'.join([prepared_name, 'lora_A', adapter_name, 'weight'])] = weights_before_mpp[name]
                elif 'lora_B' in name:
                    weights_after_mpp['.'.join([prepared_name, 'lora_B', adapter_name, 'weight'])] = weights_before_mpp[name]

    return weights_after_mpp


def prepare_lora_layers_for_attached_model(model_module, lora_scaling=None):
    """
    Prepares LoRA layers in the provided model module by configuring scaling factors
    and replacing lora layers with quantization layers that are optimized for target
    :param model_module: The PyTorch model module containing LoRA layers to be prepared.
    :param lora_scaling: Scaling factor for LoRA adapters. Can be None (no scaling), float/torch.Tensor (global scaling), or dict (mapping adapter names to scaling factors).
    Defaults to None.
    """
    for name, module in model_module.named_modules():
        if isinstance(module, PeftLoraLayer):
            for lora_adapter_name in module.lora_alpha:
                if lora_scaling is None:
                    continue
                elif isinstance(lora_scaling, (float, torch.Tensor)):
                    module.scaling[lora_adapter_name] = lora_scaling
                elif isinstance(lora_scaling, dict):
                    module.scaling = lora_scaling
                else:
                    raise ValueError(f'lora_scaling must be passed as one of [None, float, torch.Tensor, dict]! '
                                     f'Got {type(lora_scaling)} instead')

    replace_lora_layers_with_no_cast_quantizable_layers(model_module)


def _find_prepared_model_module_name(module_name, name_to_module_dict):
    for prepared_name in name_to_module_dict[module_name]:
        if prepared_name.lower().endswith("conv"):
            return prepared_name
    raise ValueError(f"Could not find a layer name ending with 'Conv' for module '{module_name}' in the name_to_module_dict.")


def _add_peft_suffix_to_encodings(
        encodings: dict[str, Any],
        full_mpp_target_module_names: set,
        verbose: bool = False,
):
    """ Adds the ".base_layer" suffix to target module names in the provided encodings.
    :encodings: The encodings dictionary containing activation or parameter encodings.
    :full_mpp_target_module_names: A set of MPP layer names to which the suffix should be added.
    :verbose: If True, print additional information during processing.
    """
    num_adapted_encodings = 0

    for encoding_name in list(encodings.keys()):
        for mpp_layer_name in full_mpp_target_module_names:
            # We assume prefix matching in encodings names
            if encoding_name.startswith(mpp_layer_name):
                new_encoding_name = encoding_name.replace(mpp_layer_name, f"{mpp_layer_name}.base_layer")
                encodings[new_encoding_name] = encodings.pop(encoding_name)

                if verbose:
                    num_adapted_encodings += 1
                    print(f"Added suffix '.base_layer' to encoding: {encoding_name} -> {new_encoding_name}")

                break

    if verbose:
        print(f"Number of adapted encodings: {num_adapted_encodings}")
