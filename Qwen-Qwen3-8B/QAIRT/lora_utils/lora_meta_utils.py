# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""This file contains utilities for generating LoRAv3 meta data"""

import json
from pathlib import Path
from typing import Union

import onnx
from aimet_torch import onnx_utils
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_common.quantsim import encoding_version as system_encoding_version


def get_updatable_tensors(
        sim: QuantizationSimModel,
        onnx_path: Union[Path, str],
        encodings_path: Union[Path, str]
):
    """
    Get updatable tensors list for a quantsim object given the onnx node to i/o tensor mapping
    :param sim: The quantsim object from which to obtain updatable tensors
    :param onnx_path: The onnx path to the model from which updatable tensors are obtained
    :param encodings_path: The encodings path to the model from which updatable tensors are obtained
    return: The updatable tensors list
    """
    updatable_tensors = set()
    encoding_names = set()

    onnx_model = onnx.load(onnx_path)

    onnx_node_to_io_tensor_map, valid_param_set = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)
    layers_to_onnx_op_names = onnx_utils.get_layers_in_io_tensor_map(onnx_node_to_io_tensor_map)

    with open(encodings_path, 'r') as f:
        model_encodings = json.load(f)

    encoding_version = model_encodings.get('version', system_encoding_version)

    if encoding_version >= '1.0.0':
        if not isinstance(model_encodings['activation_encodings'], list):
            raise TypeError(f'Expect activation encodings to be a list for {encoding_version} but got type '
                            f'{type(model_encodings["activation_encodings"])} intead!')
        for act_encoding in model_encodings['activation_encodings']:
            encoding_names.add(act_encoding['name'])
    else:
        if not isinstance(model_encodings['activation_encodings'], dict):
            raise TypeError(f'Expect activation encodings to be a dict for {encoding_version} but got type '
                            f'{type(model_encodings["activation_encodings"])} intead!')
        encoding_names.update(model_encodings['activation_encodings'].keys())

    for layer_name, qmodule in sim.named_qmodules():
        # Below code is only needed when we have different torch vs. onnx layer names
        # for original_layer_name_pattern, new_layer_name_pattern in layer_name_mappings.items():
        #     if original_layer_name_pattern in layer_name:
        #         new_layer_name = layer_name.replace(original_layer_name_pattern, new_layer_name_pattern)
        #         if new_layer_name not in encoding_names:
        #             encoding_names.remove(layer_name)
        #             encoding_names.add(new_layer_name)
        #         break

        # This will filter out all intermediate nodes without inputs (Constant node, for example). Reason is that those
        # nodes are created as intermediate inputs to other nodes. So, we don't want to double count these nodes' output
        model_input_names = set(input.name for input in onnx_model.graph.input)

        op_names = filter(
            lambda op_name: any(
                map(lambda input_name: input_name in model_input_names, onnx_node_to_io_tensor_map[op_name].inputs)
            ) or onnx_node_to_io_tensor_map[op_name].inputs != [],
            layers_to_onnx_op_names.get(layer_name, [])
        )

        # check updatable input quantizers and obtain the tensor names
        for i, input_quantizer in enumerate(qmodule.input_quantizers):
            if input_quantizer and input_quantizer._allow_overwrite:
                for op_name in op_names:
                    if i >= len(onnx_node_to_io_tensor_map[op_name].inputs):
                        continue
                    if onnx_node_to_io_tensor_map[op_name].inputs[i] in encoding_names:
                        updatable_tensors.add(onnx_node_to_io_tensor_map[op_name].inputs[i])

        # check updatable output quantizers and obtain the tensor names
        for i, output_quantizer in enumerate(qmodule.output_quantizers):
            if output_quantizer and output_quantizer._allow_overwrite:
                for op_name in op_names:
                    if i >= len(onnx_node_to_io_tensor_map[op_name].outputs):
                        continue
                    if onnx_node_to_io_tensor_map[op_name].outputs[i] in encoding_names:
                        updatable_tensors.add(onnx_node_to_io_tensor_map[op_name].outputs[i])

        # check updatable parameter quantizers and obtain the tensor names
        for param_attr, param_quantizer in qmodule.param_quantizers.items():
            param_name = f'{layer_name}.{param_attr}'
            if param_quantizer and param_quantizer._allow_overwrite and param_name in valid_param_set:
                updatable_tensors.add(param_name)

    return sorted(list(updatable_tensors))
