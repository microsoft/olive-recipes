#!/usr/bin/env python3
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""
SpinQuant R1 utilities for rotating LoRA adapter weights.

This module exposes a single helper, `apply_spinquant_r1_to_adapter`, which applies
Hadamard-based, 1/sqrt(dim)-normalized rotations to LoRA adapter weights. The rotations
are applied to selected layers detected by name substrings and to the appropriate
LoRA matrices:
- Left-hand-side (LHS) rotation on `lora_A`
- Right-hand-side (RHS) rotation on `lora_B`

For multimodal models (e.g., VLMs), you can provide separate hidden sizes and layer
filters for language and vision submodules.
"""

import torch
import copy
from aimet_torch.experimental.spinquant.hadamard_utils import get_hadamard_matrix

import logging
# Self-contained logger for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def apply_spinquant_r1_to_adapter(
    peft_state_dict,
    language_hidden_size = None,
    vision_hidden_size = None,
    language_layers_filter = "language",  # "" if LLM, "language" if VLM
    vision_layers_filter = "visual",
    layers_with_lhs_rotations = ("q_proj", "k_proj", "v_proj", "qkv", "gate_proj", "up_proj"),
    layers_with_rhs_rotations = ("o_proj", "down_proj"),
):
    """Apply SpinQuant R1 (Hadamard) rotations to LoRA adapter weights in a PEFT state dict.

    This function performs:
      * LHS rotation for `lora_A` weights of selected layers:
        `A' = (R^T @ A^T)^T`
      * RHS rotation for `lora_B` weights of selected layers:
        `B' = (B^T @ R)^T`

    The rotation matrix `R` is constructed as a Hadamard matrix normalized by `1/sqrt(dim)`,
    where `dim` is the hidden size of the respective submodule (language or vision).
    Layers are selected by matching substrings in their parameter names.

    Notes
    -----
    - The Hadamard rotation is scaled by `1/sqrt(dim)` to preserve norm.
    - Shape compatibility is assumed based on common LoRA conventions:
      `lora_A` often has shape `[out_dim, rank]` and `lora_B` has `[rank, out_dim]`.
    """

    assert language_hidden_size or vision_hidden_size, (
        "At least one of language_hidden_size and vision_hidden_size must be provided"
    )

    rotated_adapter_weights = copy.deepcopy(peft_state_dict)

    rotation_language = None
    if language_hidden_size:
        rotation_language = get_hadamard_matrix(language_hidden_size) / torch.sqrt(
            torch.tensor(language_hidden_size)
        )

    rotation_vision = None
    if vision_hidden_size:
        rotation_vision = get_hadamard_matrix(vision_hidden_size) / torch.sqrt(
            torch.tensor(vision_hidden_size)
        )

    for layername, weight in peft_state_dict.items():
        if any(sub in layername for sub in layers_with_lhs_rotations) and "lora_A" in layername:
            if rotation_language is not None and language_layers_filter in layername:
                rotated_adapter_weights.update({layername: (rotation_language.T @ weight.T).T})
            elif rotation_vision is not None and vision_layers_filter in layername:
                rotated_adapter_weights.update({layername: (rotation_vision.T @ weight.T).T})
            else:
                raise Exception(
                    f"Unable to apply left-hand-side rotation to lora_A in {layername} with weight shape {weight.shape}"
                )
        elif any(sub in layername for sub in layers_with_rhs_rotations) and "lora_B" in layername:
            if rotation_language is not None and language_layers_filter in layername:
                rotated_adapter_weights.update({layername: (weight.T @ rotation_language).T})
            elif rotation_vision is not None and vision_layers_filter in layername:
                rotated_adapter_weights.update({layername: (weight.T @ rotation_vision).T})
            else:
                raise Exception(
                    f"Unable to apply right-hand-side rotation to lora_B in {layername} with weight shape {weight.shape}"
                )
        else:
            # No rotation applied for this layer; record this for telemetry.
            logger.info('SpinQuant: no rotation applied to %s', layername)


    return rotated_adapter_weights
