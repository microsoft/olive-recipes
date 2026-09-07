#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

""" This file provides model preparation utilities """


def llm_build_preparer_converter_args(num_hidden_layers, model_input_names, separate_position_ids=True, use_qairt_mpp=False):
    '''
    This function builds converter args used in the model preparation step
    params:
     num_hidden_layers: number of hidden hidden layers
     model_input_names: list of input names of the model
    '''
    if not use_qairt_mpp:
        converter_args_param = ['--input_layout']
        converter_args_value = 'NONTRIVIAL'
        converter_args = []
        for input_param in converter_args_param:
            for input_name in model_input_names:
                if input_name == 'position_ids' and separate_position_ids:
                    converter_args += [input_param, 'position_ids[0]', converter_args_value]
                    converter_args += [input_param, 'position_ids[1]', converter_args_value]
                elif input_name == 'past_key_values':
                    for i in range(num_hidden_layers):
                        converter_args += [input_param, f'past_key_values[{i}][0]', converter_args_value]
                        converter_args += [input_param, f'past_key_values[{i}][1]', converter_args_value]
                elif input_name == 'anchor_buffer':
                    for i in range(num_hidden_layers):
                        converter_args += [input_param, f'anchor_buffer[{i}]', converter_args_value]
                elif input_name == 'keys':
                    for i in range(num_hidden_layers):
                        converter_args += [input_param, f'keys[{i}]', converter_args_value]
                else:
                    converter_args += [input_param, input_name, converter_args_value]
    else:
        from qti.aisw.tools.core.modules.converter.converter_module import InputTensorConfig
        input_tensors = []
        for input_name in model_input_names:
            if input_name == 'position_ids' and separate_position_ids:
                for i in range(2):
                    input_tensor_config=InputTensorConfig(name=f'position_ids[{i}]',
                                                        source_model_input_layout='NONTRIVIAL', source_model_input_datatype='float32')
                    input_tensors.append(input_tensor_config)
            elif input_name == 'past_key_values':
                for i in range(num_hidden_layers):
                    for j in range(2):
                        input_tensor_config=InputTensorConfig(name=f'past_key_values[{i}][{j}]',
                                                            source_model_input_layout='NONTRIVIAL', source_model_input_datatype='float32')
                        input_tensors.append(input_tensor_config)
            elif input_name == 'anchor_buffer':
                for i in range(num_hidden_layers):
                    input_tensor_config=InputTensorConfig(name=f'anchor_buffer[{i}]',
                                                        source_model_input_layout='NONTRIVIAL', source_model_input_datatype='float32')
                    input_tensors.append(input_tensor_config)
            elif input_name == 'keys':
                for i in range(num_hidden_layers):
                    input_tensor_config=InputTensorConfig(name=f'keys[{i}]',
                                                        source_model_input_layout='NONTRIVIAL', source_model_input_datatype='float32')
                    input_tensors.append(input_tensor_config)
            elif input_name == 'cache_index' or input_name == 'swa_cache_index':
                input_tensor_config=InputTensorConfig(name=input_name,
                                                    source_model_input_layout='NONTRIVIAL', source_model_input_datatype='int64')
                input_tensors.append(input_tensor_config)
            elif input_name == 'input_ids':
                input_tensor_config=InputTensorConfig(name=input_name,
                                                    source_model_input_layout='NONTRIVIAL', source_model_input_datatype='int64')
                input_tensors.append(input_tensor_config)
            else:
                input_tensor_config=InputTensorConfig(name=input_name,
                                                    source_model_input_layout='NONTRIVIAL')
                input_tensors.append(input_tensor_config)

        converter_args = {'input_tensors': input_tensors}

    return converter_args
