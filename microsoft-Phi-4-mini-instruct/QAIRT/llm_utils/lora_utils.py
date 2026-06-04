#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import re

import torch
from safetensors.torch import save_file
from aimet_torch.peft import LoraLayer

from peft.tuners.lora.layer import LoraLayer as PeftLoraLayer
from peft import PeftMixedModel
from typing import Dict

def set_lora_scaling(
        model: PeftMixedModel,
        lora_scaling: Dict[str, float]
):

    """Set the LoRA adapters' scaling parameter to the given value.

    Args:
        model (PeftModel): The model for which to set the scale parameter.
        scale (float): The scale value to set as a dictionary ['adapter_name': value].
    """
    for name, module in model.named_modules():
        if isinstance(module, PeftLoraLayer):
            for adapter_name, scaling in lora_scaling.items():
                module.scaling[adapter_name] = scaling


def save_lora_weights_after_adaptation(model: torch.nn.Module, path: str, filename_prefix: str):
    """
    Utility to save model weights after model adaptations

    :param model: PEFT model
    :param path: path where to store weights after adaptation
    :param filename_prefix: Prefix to use for filenames
    """
    param_to_name = {}

    for name, param in model.named_parameters():
        param_to_name[param] = name

    lora_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for _, param in module.lora_A.named_parameters():
                name = param_to_name[param]
                lora_weights[name] = param
            for _, param in module.lora_B.named_parameters():
                name = param_to_name[param]
                lora_weights[name] = param

    filename_prefix = filename_prefix + '.safetensor'
    model_params_path = os.path.join(path, filename_prefix)
    save_file(lora_weights, model_params_path)

def adapt_true_base_encodings(true_base_encodings_path: str, target_modules: list, output_path: str):
    """
    Modifies the encodings by appending "_base_layer" to the target module substrings
    within the file and saves the updated content.
    :param true_base_encodings_path (str): Path to the encodings file.
    :param target_modules (list): List of module names to search for and modify.
    :param output_path (str): Path to save the modified encodings.
    """
    # Read the content of the file
    with open(true_base_encodings_path, 'r') as file:
        lines = file.readlines()
    # Initialize modified lines
    modified_lines = []
    for line in lines:
        modified_line = line
        for module in target_modules:
            if module in line:
                # Append ".base_layer" after the module substring
                modified_line = modified_line.replace(module, f"{module}.base_layer")
        modified_lines.append(modified_line)
    # Write the modified lines back to the output file
    with open(output_path, 'w') as file:
        file.writelines(modified_lines)
 
def adapt_peft_config_for_lora_meta_data(adapter_name, peft_config):
    """
    Given the target_modules in the HuggingFace PEFT config, find all the corresponding full layer names
    in the JSON prepare layer map. Return a list of the new target modules.
    :param adapter_name (str): Name of the adapter.
    :param peft_config (dict): Individual HuggingFace PEFT config containing `target_modules`.
    Returns:
        set: A list of new target modules (full prepared graph layer names matching the criteria).
    """
    mpp_peft_config = {}
    mpp_peft_config["name"] = adapter_name
    mpp_peft_config["rank"] = peft_config.__dict__["r"]
    mpp_peft_config["alpha"] = peft_config.__dict__["lora_alpha"]
    mpp_peft_config["target_modules"] = list(peft_config.target_modules)
    return mpp_peft_config


def convert_linear_to_conv_weights(weights_dict):
    """
    Convert 2D linear LoRA weights to 4D format compatible with convolutional operations.
    Processes a state dict and returns the converted version.
    :param weights_dict: Dictionary containing LoRA weights
    Returns:
        Dictionary with weights converted to convolutional format
    """
    converted_weights = {}
    for name, tensor in weights_dict.items():
        if 'lora_A' in name or 'lora_B' in name:
            converted_weights[name] = tensor.unsqueeze(-1).unsqueeze(-1)
        else:
            converted_weights[name] = tensor
    return converted_weights

