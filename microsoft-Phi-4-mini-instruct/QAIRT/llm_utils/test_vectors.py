#!/usr/bin/env python3
# ==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2024 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# ==============================================================================
""" utils to generate test vectors """
from typing import Tuple, Union, List, Dict
import re
import os

import numpy as np
import torch.nn
import torch
import pickle
import contextlib

from tqdm import tqdm
from aimet_torch.utils import change_tensor_device_placement, nested_map

from aimet_torch.utils import in_eval_mode, is_leaf_module
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.layer_output_utils import LayerOutput, LayerOutputUtil, NamingScheme

from aimet_torch.v2.nn.base import BaseQuantizationMixin
from aimet_torch.v2.quantization import QuantizedTensorBase
from aimet_torch.quantsim import ExportableQuantModule

MODULE_TYPE_FOR_ATTACHING_HOOK = (ExportableQuantModule,)
modules_to_treat_as_leaf = []

def to_torch_tensor(t):
    """ utilty to move test vectors from DequantizedTensor to torch.Tensor """
    return nested_map(t, lambda x: torch.tensor(x) if isinstance(x, QuantizedTensorBase) else x)

def to_cpu(t):
    return change_tensor_device_placement(t, torch.device('cpu'))

def quantizers_state(sim, disabled) -> contextlib.ExitStack:
    exit_stack = contextlib.ExitStack()
    if disabled:
        for _, module in sim.model.named_modules():
            if isinstance(module, BaseQuantizationMixin):
                exit_stack.enter_context(module._remove_all_quantizers())
    return exit_stack

def run_hook_for_layers_with_given_input_get_output(model: torch.nn.Module,
                                                    input_tensor: Union[torch.Tensor, Tuple, Dict], hook,
                                                    module_type_for_attaching_hook=None, module_regex_to_include=None,
                                                    leaf_node_only=True, fwd_func=None):
    """
    Register the given hook function for all layers in the model
    :param model: Model
    :param input_tensor: Input tensor to the model. If more than one model inputs, use a tuple
    :param hook: Hook function to register
    :param module_type_for_attaching_hook: Tuple of torch.nn module types for which hook has to be attached
    :param leaf_node_only: Set to False if all modules are required
    :param fwd_func: forward function for model inference
    :return: None
    """
    # ------------------------
    # Register hook function
    # ------------------------
    hooks = []
    # All leaf modules
    modules = []

    # Based on the modules in modules_to_treat_as_leaf, we do not want to further continue searching for next level
    # of modules present in modules_to_treat_as_leaf. To achieve this, save them in modules_to_skip
    modules_to_skip = set()

    if module_regex_to_include:
        patterns = [re.compile(pattern) for pattern in module_regex_to_include]
        name_match_modules = [module for name, module in model.named_modules() if  any (re.match(pattern, name) for pattern in patterns)]
    else:
        name_match_modules = model.modules()

    for module in name_match_modules:
        if module not in modules_to_skip:
            # pylint: disable=protected-access
            if isinstance(module, tuple(modules_to_treat_as_leaf)):
                modules.append(module)
                # check for modules inside the 'module' and add them to modules_to_skip
                for sub_module in module._modules.values():
                    modules_to_skip.add(sub_module)
            else:
                if leaf_node_only:
                    if is_leaf_module(module):
                        modules.append(module)
                else:
                    modules.append(module)

    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [module for module in modules if isinstance(module, module_type_for_attaching_hook)]

    try:
        for module in modules:
            hooks.append(module.register_forward_hook(hook))

        # ------------------------------------------------
        # Run forward pass to execute the hook functions
        # ------------------------------------------------
        with in_eval_mode(model), torch.no_grad():
            if fwd_func:
                output = fwd_func(model, input_tensor)
            else:
                if isinstance(input_tensor, (list, tuple)):
                    output = model(*input_tensor)
                elif isinstance(input_tensor, dict):
                    output = model(**input_tensor)
                else:
                    output = model(input_tensor)

    finally:
        # --------------------------
        # Remove all hooks we added
        # --------------------------
        for h in hooks:
            h.remove()

        return output


class LLMLayerOutput(LayerOutput):
    def __init__(self, model: torch.nn.Module, dir_path: str, naming_scheme: NamingScheme = NamingScheme.PYTORCH,
                 dummy_input = None, onnx_export_args: Union[OnnxExportApiArgs, Dict] = None, regex_patterns = None):
        super().__init__(model, dir_path, naming_scheme, dummy_input, onnx_export_args)
        self.regex_patterns = regex_patterns

    def record_outputs(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        """
        Hook function to capture output of a layer.

        :param module: Layer-module in consideration.
        :param input: Input of the layer-module.
        :param output: Output of the layer-module.
        :return: None
        """
        layer_name = self.module_to_name_dict[module]
        self.layer_name_to_layer_output_dict[layer_name] = {"input": to_cpu(input[0]), "output": to_cpu(output)}

    def get_outputs(self, input_batch) -> Dict[str, torch.Tensor]:
        """
                This function captures layer-outputs and renames them as per the AIMET exported pytorch/onnx/torchscript model.

                :param input_batch: Batch of inputs for which we want to obtain layer-outputs.
                :return: layer-name to layer-output batch dict
                """

        # Fetch outputs of all the layers
        self.layer_name_to_layer_output_dict = {}
        if self.is_quantsim_model:
            # Apply record-output hook to QuantizeWrapper modules (one node above leaf node in model graph)
            model_output = run_hook_for_layers_with_given_input_get_output(self.model, input_batch, self.record_outputs,
                                                                    module_type_for_attaching_hook=MODULE_TYPE_FOR_ATTACHING_HOOK,
                                                                    leaf_node_only=False, module_regex_to_include=self.regex_patterns)
        else:
            # Apply record-output hook to Original modules (leaf node in model graph)
            model_output = run_hook_for_layers_with_given_input_get_output(self.model, input_batch, self.record_outputs,
                                                                    leaf_node_only=True, module_regex_to_include=self.regex_patterns)

        # Rename outputs according to pytorch/onnx/torchscript model
        layer_output_name_to_layer_output_dict = LayerOutput.rename_layer_outputs(self.layer_name_to_layer_output_dict,
                                                                                  self.layer_name_to_layer_output_name_dict)

        return layer_output_name_to_layer_output_dict, model_output


class LLMLayerOutputUtil(LayerOutputUtil):
    def __init__(self, model: torch.nn.Module, dir_path: str, file_prefix: str,  naming_scheme: NamingScheme = NamingScheme.PYTORCH,
                 dummy_input = None, onnx_export_args: Union[OnnxExportApiArgs, Dict] = None, regex_patterns = None):
        """
        Constructor for LayerOutputUtil.

        :param model: Model whose layer-outputs are needed.
        :param dir_path: Directory wherein layer-outputs will be saved.
        :param naming_scheme: Naming scheme to be followed to name layer-outputs. There are multiple schemes as per
            the exported model (pytorch, onnx or torchscript). Refer the NamingScheme enum definition.
        :param dummy_input: Dummy input to model. Required if naming_scheme is 'NamingScheme.ONNX' or 'NamingScheme.TORCHSCRIPT'.
        :param onnx_export_args: Should be same as that passed to quantsim export API to have consistency between
            layer-output names present in exported onnx model and generated layer-outputs. Required if naming_scheme is
            'NamingScheme.ONNX'.
        """
        super().__init__(model, dir_path, naming_scheme, dummy_input, onnx_export_args)
        self.output_dir = dir_path
        self.file_prefix = file_prefix

        # Utility to capture layer-outputs
        self.layer_output = LLMLayerOutput(model=model, naming_scheme=naming_scheme, dir_path=dir_path, dummy_input=dummy_input,
                                            onnx_export_args=onnx_export_args, regex_patterns=regex_patterns)

    def generate_layer_outputs(self, input_batch, batch_idx):
        """
        This method captures output of every layer of a model & saves the inputs and corresponding layer-outputs to disk.

        :param input_batch: Batch of inputs for which we want to obtain layer-outputs.
        :return: None
        """

        # Obtain layer-output name to output dictionary
        layer_output_batch_dict, model_outputs = self.layer_output.get_outputs(input_batch)

        test_vectors = {f"{batch_idx}": {**to_cpu(to_torch_tensor(input_batch)),
                                         **to_cpu(to_torch_tensor(layer_output_batch_dict))}}

        assert os.path.exists(self.output_dir), "output_dir for test vectors doesn't exist"

        for key, value in test_vectors.items():
            filename = os.path.join(self.output_dir, self.file_prefix + f"_{batch_idx}.pkl")
            with open(filename, 'wb') as file:
                pickle.dump({key: value}, file)

        return model_outputs

def generate_test_vectors(sim, forward_pass_manager, data_loader, output_dir, num_batches, test_vector_layers, input_names):
    vector_output_dir = os.path.join(output_dir, "test_vectors")
    os.makedirs(vector_output_dir, exist_ok=True)

    def _sanitize_and_update_test_vectors(test_vectors, test_outputs):
        if "past_key_values" in test_outputs:
            test_outputs["output_key_values"] = test_outputs.pop("past_key_values")

        test_vectors.update(to_cpu(test_outputs))

        if "lm_logits" in test_vectors:
            test_vectors["logits"] = test_vectors.pop("lm_logits")

        past_key_values = []
        for i in range(forward_pass_manager.num_layers):
            past_key = test_vectors.pop(f"past_key_{i}_in")
            past_val = test_vectors.pop(f"past_value_{i}_in")
            past_key_values.append([past_key, past_val])
        test_vectors["past_key_values"] = past_key_values

    for idx, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Test vector generation")):
        if idx >= num_batches:
            break
        for vector_type in ['fp', 'qt']:

            recorder = LLMLayerOutputUtil(forward_pass_manager.model, dir_path=vector_output_dir,
                                                           file_prefix=vector_type, regex_patterns=test_vector_layers)

            with quantizers_state(sim, disabled=(vector_type == 'fp')):
                prepared_inputs, kvcache_info_bundle = forward_pass_manager.prepare_inputs(
                    input_ids=batch['input_ids'][..., :forward_pass_manager.num_tokens], input_names= input_names)
                outputs = recorder.generate_layer_outputs(prepared_inputs, idx)
                outputs = forward_pass_manager.prepare_outputs(outputs, prepared_inputs, kvcache_info_bundle)

            filename = os.path.join(vector_output_dir, f"{vector_type}_{idx}.pkl")
            test_vector_dict = np.load(filename, allow_pickle=True)

            _sanitize_and_update_test_vectors(test_vector_dict[f"{idx}"], outputs)
            test_vector_dict = to_cpu(to_torch_tensor(test_vector_dict))
            
            with open(filename, 'wb') as file:
                pickle.dump(test_vector_dict, file)
