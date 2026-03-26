#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch
from typing import Dict, Any
from genai_lib.common.onnxruntime_utils import OutputBufferCreator
from transformers import PretrainedConfig

def map_llama_onnx_flattened_name_to_tensor(onnx_name: str,
                            tensor_dict: Dict[str, Any],
                            is_output: bool,
                            delim: str = "_") -> torch.Tensor:
    """
    Maps a flattened ONNX input/output name to the corresponding PyTorch tensor in a condensed prepared_inputs/output_buffer, for llama-like models.

    params:
        onnx_name (str): The flattened name of the ONNX input/output.
        tensor_dict (Dict[str, Any]): The prepared_inputs/output_buffer.
        is_output (bool): Whether the ONNX name is an output. (Unused for llama-like models)
        delim (str): The delimiter used in the flattened name.
    returns:
        torch.Tensor: The corresponding tensor in the prepared_inputs/output_buffer.
    """
    parts = onnx_name.split(delim)
    if parts[0] == "position":
        return tensor_dict["position_ids"][0 if parts[2] == "cos" else 1]
    
    if parts[0] == "past":
        # past_{key/value}_{layer_idx}_{in/out}
        kv_idx = 1 if parts[1] == "value" else 0
        return tensor_dict["past_key_values"][int(parts[2])][kv_idx]
    
    return tensor_dict[onnx_name]

class LlamaOutputBufferCreator(OutputBufferCreator):
    """
    Allocates an empty output buffer for a vanilla Llama-like model.

    params:
        batch_size (int): The batch size.
        max_input_tokens (int): The max input tokens.
        config (PretrainedConfig): The model configuration.
        device (torch.device): The device to allocate the output buffer.
    """
    def __init__(
            self,
            batch_size:int, 
            max_input_tokens:int, 
            config:PretrainedConfig, 
            device:torch.device):
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.config = config
        self.device = device
    
    def create_buffer(self) -> Dict[str, Any]:
        """
        Method to build the output buffer assuming logits and past_key_values are the only outputs.

        returns:
            Dict[str, Any]: The output buffer.
        """
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        output = {
            "logits": torch.empty(self.batch_size, self.max_input_tokens, self.config.vocab_size, device=self.device, dtype=torch.float32).contiguous(),
            "past_key_values": tuple((
            torch.empty(self.batch_size, self.config.num_key_value_heads, head_dim, self.max_input_tokens, device=self.device,
                            dtype=torch.float32).contiguous(),
            torch.empty(self.batch_size, self.config.num_key_value_heads, self.max_input_tokens,head_dim, device=self.device,
                            dtype=torch.float32).contiguous()) for _ in range(self.config.num_hidden_layers))
        }
        return output