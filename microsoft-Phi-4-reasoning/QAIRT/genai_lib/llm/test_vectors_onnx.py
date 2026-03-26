#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import pickle
import re
from typing import List, Dict, Tuple, Set, Optional, Any, Sequence
import numpy as np

from contextlib import contextmanager, ExitStack

import torch
from torch.utils._pytree import tree_map_only

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.layer_output_utils import LayerOutput

from aimet_common.layer_output_utils import save_layer_output_names

from genai_lib.common.onnxruntime_utils import ONNXNameMapper, OutputBufferCreator, ORTInferenceModule

from onnx import ModelProto

@contextmanager
def disable_quantizers(sim, quantizer_names: Set[str]):
    """
    Disables all quantizers in quantizer_names inside the context

    Copied from AIMET-ONNX repo: https://github.qualcomm.com/qualcomm-ai/aimet/blob/develop/TrainingExtensions/onnx/src/python/aimet_onnx/utils.py
    """
    if not isinstance(quantizer_names, set):
        quantizer_names = set(quantizer_names)

    if not quantizer_names.issubset(sim.qc_quantize_op_dict.keys()):
        raise RuntimeError(
            f"quantizer_names contains non-existent quantizers: {quantizer_names - sim.qc_quantize_op_dict.keys()}"
        )

    is_enabled = {
        name: sim.qc_quantize_op_dict[name].enabled for name in quantizer_names
    }

    try:
        for name in quantizer_names:
            sim.qc_quantize_op_dict[name].enabled = False

        yield

    finally:
        for name in quantizer_names:
            sim.qc_quantize_op_dict[name].enabled = is_enabled[name]


class LLMLayerOutput:
    """
    This class creates a dictionary of node name to node-input/output, and model-input name to model-input. It also produces model output.
    """

    def __init__(self,
                 model: ModelProto,
                 providers: Sequence[str | Tuple[str, Dict[Any, Any]]],
                 dir_path: str,
                 onnx_name_to_tensor: ONNXNameMapper,
                 output_buffer_creator: OutputBufferCreator,
                 regex_patterns:Optional[List[str]] = None):
        """
        Constructor - Initializes lists required for capturing and naming layer-outputs

        params:
            model (ModelProto): Onnx Model
            providers (List): List of onnxruntime execution providers
            dir_path (str): directory path to save the layer-inputs/outputs
            onnx_name_to_tensor (ONNXNameMapper): ONNXNameMapper object which maps between the onnx names and pytorch tensors
            output_buffer_creator (OutputBufferCreator): OutputBufferCreator object that creates an Output Buffer for IO Binding
            regex_patterns (Optional[List[str]]): list of regex patterns to match the layer names
        """
        self.model = model
        self.output_buffer_creator = output_buffer_creator
        self.onnx_name_to_tensor = onnx_name_to_tensor
        self.original_output_names = [output.name for output in model.graph.output]

        self.node_name_to_io_names = {}

        # retrieve list of inputs to nodes, list of outputs of nodes, a dict of node name to inputs/outputs, and a list of all the activations (regardless of regex matching)
        self.node_input_activation_names, self.node_output_activation_names, self.node_name_to_io_names, all_layer_output_names = LLMLayerOutput.get_activation_names(
            self.model, regex_patterns)
        # mark any activations that are quantized
        quantized_activation_names = [
            n for n in self.node_input_activation_names + self.node_output_activation_names if n.endswith("_updated")
        ]
        # add the the non-quantized versions of those activations to the list to remove 
        # remove the "_updated" at the end of  the quantized activation name to get the original
        activations_to_remove = {n[:-8] for n in quantized_activation_names}

        # add any qdq activations to the list to remove as well
        activations_to_remove.update(
            n for n in (self.node_input_activation_names + self.node_output_activation_names) if n.endswith("_qdq")
            )

        # remove in-place the activations marked earlier, by changing the values inside the list, while keeping the list in-place
        for lst in (self.node_input_activation_names, self.node_output_activation_names):
            lst[:] = [n for n in lst if n not in activations_to_remove]

        # remove those activations from the node_name_to_io_names dict as well
        for key in self.node_name_to_io_names:
            for io in ["input", "output"]:
                self.node_name_to_io_names[key][io] = [name for name in self.node_name_to_io_names[key][io] if
                                                       name not in activations_to_remove]

        # recompile a full list of activations to hook
        self.activation_names = self.node_input_activation_names + self.node_output_activation_names

        # ONNX "hook" for activations
        LayerOutput.register_activations(self.model, self.activation_names)

        # Build ONNX runtime inference session
        self.session = QuantizationSimModel.build_session(self.model, providers)

        # Reverting the added outputs until the output length matches the original state
        # This does not affect the inference session, as self.session does not have a live reference to self.model.
        while len(self.model.graph.output) > len(self.original_output_names):
            self.model.graph.output.pop()

        # Sanitize layer output names to save
        sanitized_all_layer_output_names = [LLMLayerOutput.sanitize_activation_name(name) for name in all_layer_output_names if not (name.endswith("_updated") or name.endswith("qdq"))]

        # Save all model activations in topological order of model graph, for use when comparing layer-outputs
        # NOTE: This only saves layer output names, not layer input names, to match behavior of torch-based test vectors.
        save_layer_output_names(sanitized_all_layer_output_names, dir_path)

    @staticmethod
    def sanitize_activation_name(activation_name: str) -> str:
        """
        This function sanitizes the activation name by replacing non-alphanumeric characters with underscores.

        params:
            activation_name (str): activation name
        returns:
            str: sanitized activation name
        """
        return re.sub(r"\W+", "_", activation_name.replace("_updated", "")).strip("_")

    @staticmethod
    def get_activation_names(model: ModelProto, regex_patterns) -> Tuple[
        List[str], List[str], Dict[str, Dict[str, List[str]]], List[str]]:
        """
        This function fetches the activation names (model_input, node_input, node_output names) of the given onnx model, that match the provided regex patterns.
        
        params:
            model (ModelProto): ONNX model
            regex_patterns (list): list of regex patterns
        return:
            tuple: Tuple containing lists of activation names for node inputs and node outputs, a Dict of node name to Dicts of node input/output names, and a list of all activation names
        """
        patterns = [re.compile(pattern) for pattern in (regex_patterns if regex_patterns is not None else [])]

        node_name_to_io_names = {}
        node_input_activation_names = []
        node_output_activation_names = []
        # we build a list of all_activation names for save_layer_output_names function
        all_layer_output_names = []

        for node in model.graph.node:
            # Regardless of regex expression matching, collect layer output name to save
            for output_name in node.output:
                all_layer_output_names.append(output_name)
            # Ensure that node is one of the desired nodes
            if not any(re.match(pattern, node.name) for pattern in patterns):
                continue
            # Construct node name dict
            node_name_to_io_names[node.name] = {"input": [], "output": []}
            # Add each input name to the list and dict
            for input_name in node.input:
                node_input_activation_names.append(input_name)
                node_name_to_io_names[node.name]["input"].append(input_name)
            # Add each output name to the list and dict 
            for output_name in node.output:
                node_output_activation_names.append(output_name)
                node_name_to_io_names[node.name]["output"].append(output_name)
        return node_input_activation_names, node_output_activation_names, node_name_to_io_names, all_layer_output_names

    def get_outputs(self, input_dict: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Tuple]], Dict[str, torch.Tensor]]:
        """
        This function creates node-input/output dict, and also returns model output

        params:
            input_dict (Dict[str, Any]): Dictionary that contains inputs to model
        returns:
            Tuple[Dict[str, Dict[str, Tuple]], Dict[str, torch.Tensor]]: Tuple of node input/output dict and model output
        """

        device = "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"
        
        # Ensure input tensors are contiguous
        input_dict = tree_map_only(torch.Tensor, lambda t: t.contiguous(), input_dict)

        # Create output buffer
        output_buffer = self.output_buffer_creator.create_buffer()

        # Bind the original outputs to the output buffer
        io_binding = ORTInferenceModule.bind_io(session=self.session, 
                                                onnx_name_to_tensor=self.onnx_name_to_tensor, 
                                                input_buffer = input_dict, 
                                                output_buffer=output_buffer, 
                                                output_names = self.original_output_names)

        # Bind all the newly hooked outputs
        for output_name in self.activation_names:
            io_binding.bind_output(output_name, device)

        # Inference
        self.session.run_with_iobinding(io_binding)

        # This produces a list of the outputs, first the original outputs, and then the newly hooked outputs
        all_outputs = io_binding.copy_outputs_to_cpu()

        # Fix the start index
        start_idx = len(self.original_output_names)

        # Zip the node input values
        node_input_values_dict = dict(zip(self.node_input_activation_names,
                                          all_outputs[start_idx:start_idx + len(self.node_input_activation_names)]))
        
        # Zip the node output values
        start_idx += len(self.node_input_activation_names)
        node_output_values_dict = dict(zip(self.node_output_activation_names, all_outputs[start_idx:]))

        # Initialize layer name to layer io dict
        layer_name_to_layer_io_dict = {}

        # Create {"input": (input_values), "output":  (output_values)} Entries
        for node_name, node_io_names in self.node_name_to_io_names.items():
            # Sanitize each name individually
            sanitized_node_name = LLMLayerOutput.sanitize_activation_name(node_name)
            layer_name_to_layer_io_dict[sanitized_node_name] = {"input": [], "output": []}

            input_values = []
            for input_name in node_io_names["input"]:
                input_values.append(node_input_values_dict[input_name])
            output_values = []
            for output_name in node_io_names["output"]:
                output_values.append(node_output_values_dict[output_name])
            layer_name_to_layer_io_dict[sanitized_node_name] = {"input": tuple(input_values),
                                                                "output": tuple(output_values)}

        return layer_name_to_layer_io_dict, output_buffer


class LLMLayerOutputUtil:
    """Class to capture and save inputs and outputs of intermediate nodes of an onnx model"""

    def __init__(
            self,
            model: ModelProto,
            dir_path: str,
            file_prefix: str,
            providers: Sequence[str | Tuple[str, Dict[Any, Any]]],
            onnx_name_to_tensor: ONNXNameMapper,
            output_buffer_creator: OutputBufferCreator,
            regex_patterns:Optional[List[str]]=None):
        """
        Constructor - It initializes the utility classes that captures and saves the layer-inputs/outputs of an onnx model

        params:
            model (ModelProto): Onnx Model
            dir_path (str): directory path to save the layer-inputs/outputs
            file_prefix (str): file prefix to save the layer-inputs/outputs
            onnx_name_to_tensor (ONNXNameMapper): ONNXNameMapper object which maps between the onnx names and pytorch tensors
            output_buffer_creator (OutputBufferCreator): OutputBufferCreator object that creates an Output Buffer for IO Binding
            device (int): device id to run the model on
            regex_patterns (Optional[List[str]]): list of regex patterns to match the layer names
        """
        self.output_dir = dir_path
        self.file_prefix = file_prefix

        self.layer_output = LLMLayerOutput(model=model, 
                                           providers=providers, 
                                           dir_path=dir_path, 
                                           onnx_name_to_tensor=onnx_name_to_tensor,
                                           output_buffer_creator=output_buffer_creator,
                                           regex_patterns=regex_patterns)

    def generate_layer_outputs(self, input_dict: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        This function captures input/output of model, as well as every node that has been matched with the regex patterns. Then it saves these values to disk.
        
        params:
            input_batch (Dict[str, Any]): input batch to the model
            batch_idx (int): batch index
        returns:
            Dict[str, torch.Tensor]: dictionary containing the original model outputs
        """
        layer_output_dict, model_outputs = self.layer_output.get_outputs(input_dict)

        # Convert to torch to match torch-based test vectors export
        test_vectors = {f"{batch_idx}": tree_map_only(np.ndarray, torch.from_numpy, {**input_dict, **layer_output_dict})}

        assert os.path.exists(self.output_dir), "output_dir for test vectors doesn't exist"

        for key, value in test_vectors.items():
            filename = os.path.join(self.output_dir, f"{self.file_prefix}_{batch_idx}.pkl")
            with open(filename, 'wb') as file:
                pickle.dump({key: value}, file)

        return model_outputs


def generate_test_vectors(sim: QuantizationSimModel, model_inputs: Dict[str, Any], output_dir: str,
                          batch_index: int, test_vector_layers: List[str], onnx_name_to_tensor: ONNXNameMapper, output_buffer_creator: OutputBufferCreator):
    """
    This function captures inputs/outputs to the model and also inputs/outputs to the nodes that are matched by the provided list of regex patterns.
    It does this for both Floating point and Quantized models and saves the values to disk.

    params:
        sim (QuantizationSimModel): QuantizationSimModel object
        model_inputs (Dict[str, Any]): input batch to the model
        output_dir (str): output directory to save test vectors
        batch_index (int): batch index
        test_vector_layers (List[str]): list of regex patterns to match the nodes to capture the layer-inputs/outputs
        onnx_name_to_tensor (ONNXNameMapper): ONNXNameMapper object which maps between the onnx names and pytorch tensors
        output_buffer_creator (OutputBufferCreator): OutputBufferCreator object that creates an Output Buffer for IO Binding
    returns:
        None
    """
    vector_output_dir = os.path.join(output_dir, "test_vectors")
    os.makedirs(vector_output_dir, exist_ok=True)

    def _convert_and_update_test_vectors(test_vectors, test_outputs):
        if "past_key_values" in test_outputs:
            test_outputs["output_key_values"] = test_outputs.pop("past_key_values")

        test_vectors.update(test_outputs)

    for vector_type in ['fp', 'qt']:
        recorder = LLMLayerOutputUtil(sim.model.model, 
                                      dir_path=vector_output_dir, 
                                      file_prefix=vector_type,
                                      providers=sim.providers,
                                      onnx_name_to_tensor=onnx_name_to_tensor,
                                      output_buffer_creator=output_buffer_creator,
                                      regex_patterns=test_vector_layers)

        ctx_managers = ExitStack()
        if vector_type == 'fp':
            ctx_managers.enter_context(disable_quantizers(sim, sim.qc_quantize_op_dict.keys()))

        with ctx_managers:
            model_outputs = recorder.generate_layer_outputs(model_inputs, batch_index)

        filename = os.path.join(vector_output_dir, f"{vector_type}_{batch_index}.pkl")
        test_vector_dict = np.load(filename, allow_pickle=True)

        _convert_and_update_test_vectors(test_vector_dict[f"{batch_index}"], model_outputs)

        with open(filename, 'wb') as file:
            pickle.dump(test_vector_dict, file)

        # Delete the inference session manually to free up memory
        recorder.layer_output.session = None
        del recorder
        torch.cuda.empty_cache()