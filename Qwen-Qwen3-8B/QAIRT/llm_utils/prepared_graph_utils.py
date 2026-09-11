#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Utility methods related to the static graphs"""
import re
from safetensors.torch import load_file, save_file
import os
import torch
import shutil
import warnings

def replace_arn_to_arx(input_dir, output_dir, current_ar, desired_ar, prepare_filename, cache_index_tensor_list=None, current_cl=None, desired_cl=None):
    """
    This function converts the prepared graph from current_ar->desired_ar avoiding the need to re-prepare the graph in NB1

    This utility assumes that the current_ar is a magic number (a number which will not appear in the model) otherwise it might interfere with the model's hidden_size, dimensions.

    We assume that the prepared artifacts are stored inside the prepare folder within the `current_ar_output_dir` and the name would be `prepare_filename`
    :param input_dir: directory where the current ar prepared artifacts are stored
    :param output_dir: the output dir path (inclusive of /prepare) to which the .py and .safetensors corresponding to the desired_ar are written
    :param current_ar: the current AR (autoregressive) value
    :param desired_ar: the desired AR value
    :param prepare_filename: name of the .py/ .safetensors file
    :param cache_index_tensor_list: list of tensor names in the prepared pytorch graph which are of AR size, hence need to be updated.
    :param current_ar: the current context_length value
    :param desired_ar: the desired context_length value
    Note: current logic assumes that these tensors are torch.arange(AR), if we get a new tensor which is not arange but related to AR, this logic will need custom handling.g
    """
    warnings.warn(f"generated a candidate ARN-{desired_ar} variant model by constant substitution in source code, please verify for correctness of generated model")

    # Get the current ar prepared paths
    prepared_py_path_current_ar = os.path.join(input_dir, f'{prepare_filename}.py')
    prepared_ckpt_path_current_ar = os.path.join(input_dir, f'{prepare_filename}.safetensors')

    # paths for storing the desired ar artifacts
    os.makedirs(output_dir, exist_ok=True)
    prepared_py_path_desired_ar = os.path.join(output_dir, f'{prepare_filename}.py')
    prepared_ckpt_path_desired_ar = os.path.join(output_dir, f'{prepare_filename}.safetensors')

    ######## Step1: Update the `current_ar` in .py file to `desired_ar` using regex ##########
    with open(prepared_py_path_current_ar, "r") as file:
        content = file.read()

    pattern = rf'(?<!\d){current_ar}(?!\d)'
    updated_content = re.sub(pattern, str(desired_ar), content)

    if current_cl:
        assert desired_cl, "desired CL needs to specified for successful `current_cl -> desired_cl` conversion"
        pattern = rf'(?<!\d){current_cl}(?!\d)'
        updated_content = re.sub(pattern, str(desired_cl), updated_content)

    with open(prepared_py_path_desired_ar, "w") as file:
        file.write(updated_content)

    ######## Step2: Update the safetensors file if necessary and write to the output dir  ##########
    if cache_index_tensor_list:
        checkpoint = load_file(prepared_ckpt_path_current_ar)

        # since we create the cache_tensor inside the model which is of length current_ar, so we need to update the weights in the state_dict to reflect the desired_ar.
        for tensor_name in cache_index_tensor_list:
            if tensor_name in checkpoint.keys():
                checkpoint[tensor_name] = torch.arange(desired_ar, dtype=checkpoint[tensor_name].dtype, device=checkpoint[tensor_name].device)

        save_file(checkpoint, prepared_ckpt_path_desired_ar)
