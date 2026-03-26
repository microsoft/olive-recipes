#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch
from torch.utils._pytree import tree_map_only

import numpy as np

import onnxruntime as ort

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Optional, List

class ONNXNameMapper(Protocol):
    """
    Callable signature that maps a flattened ONNX input/output name to the 
    corresponding PyTorch tensor stored in an input/ouput dict.
    
    A function that matches this protocol is responsible for looking up the
    correct tensor when we require a memory pointer for I/O-binding. 

    params:
        onnx_name (str): The flattened name of the ONNX input/output.
        tensor_dict (Dict[str, Any]): The prepared_inputs/output_buffer.
        is_output (bool): Whether the ONNX name is an output, in case different logic is necessary.
        delim (str): The delimiter used in the flattened name.
    returns:
        torch.Tensor: The corresponding tensor in the prepared_inputs/output_buffer.
    """
    def __call__(
            self,
            onnx_name: str,
            tensor_dict: Dict[str, Any],
            is_output: bool,
            delim: str = "_"
    ) -> torch.Tensor:
        ...

class OutputBufferCreator(ABC):
    """
    Interface that builds an output buffer for ONNX Runtime IO Binding. 

    Every concrete implementation can customize the parameters required 
    in its constructor. Caller then keeps a reference to the instance and invokes
    create_buffer() or calls the instance itself, with no need for arguments.
    """
    @abstractmethod
    def create_buffer(self) -> Dict[str, Any]:
        """
        Allocate and return the buffer that will hold the model outputs.
        """
        ...
    
    def __call__(self) -> Dict[str, Any]:
        return self.create_buffer()

class ORTInferenceModule(torch.nn.Module):
    """
    A PyTorch module that wraps an ONNX Runtime inference session. It applies I/O binding and creates an output buffer.

    params:
        ort_session (ort.InferenceSession): The ONNX Runtime inference session.
        device (torch.device): device to place module on
        onnx_name_to_tensor (ONNXNameMapper): A function that maps flattened ONNX names (str) to PyTorch Tensors.
        prepare_output_buffer (Callable[..., Dict[str, Any]]): A function that prepares the output buffer.
    """
    def __init__(self, 
                 ort_session:ort.InferenceSession, 
                 device: torch.device,
                 onnx_name_to_tensor: ONNXNameMapper, 
                 output_buffer_creator: OutputBufferCreator):
        """
        Constructor
        """
        super().__init__()
        self.session = ort_session

        self.onnx_name_to_tensor = onnx_name_to_tensor

        self.output_buffer_creator = output_buffer_creator

        # register a single dummy parameter so Transformers pipeline() is able to know where the module is located
        self.register_parameter("_dummy", torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False))
    
    @staticmethod
    def bind_io(session:ort.InferenceSession,
                onnx_name_to_tensor: ONNXNameMapper, 
                 input_buffer: Dict[str, Any], 
                 output_buffer: Dict[str, Any],
                 output_names: Optional[List[str]] = None) -> ort.IOBinding:
        """
        This function produces an IOBinding that binds the input buffer and output buffer to the ONNX Runtime Session.

        params:
            session: (ort.InferenceSession): ONNX Runtime inference session to create IO Binding for
            onnx_name_to_tensor (ONNXNameMapper): A function that maps flattened ONNX names (str) to PyTorch Tensors.
            input_buffer (Dict[str, Any]): Dict that contains prepared_inputs.
            output_buffer (Dict[str, Any]): Dict that contains empty memory allocated for outputs.
            output_names (Optional[List[str]]): List of output names, in case user doesn't want to bind all outputs of the session

        returns:
            ort.IOBinding: IOBinding that binds the input buffer and output buffer to the ONNX Runtime Session
        """
        io_binding = session.io_binding()
        pt_to_np = {
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.float32": np.float32,
            "torch.float16": np.float16
        }

        for input in session.get_inputs():
            # Retrive the tensor corresponding to the ONNX Runtime input name from the input_buffer
            tensor_to_bind = onnx_name_to_tensor(input.name, input_buffer, is_output=False)
            io_binding.bind_input(
                name = input.name,
                device_type = tensor_to_bind.device.type,
                device_id = 0 if tensor_to_bind.device.type == "cpu" else tensor_to_bind.device.index,
                element_type = pt_to_np[repr(tensor_to_bind.dtype)],
                shape = tuple(tensor_to_bind.shape),
                buffer_ptr = tensor_to_bind.data_ptr()
            )

        output_names = (
            output_names
            if output_names is not None
            else [o.name for o in session.get_outputs()]
        )

        for output in output_names:
            # Retrive the tensor corresponding to the ONNX Runtime output name from the output_buffer
            tensor_to_bind = onnx_name_to_tensor(output, output_buffer, is_output=True)
            io_binding.bind_output(
                name = output,
                device_type = tensor_to_bind.device.type,
                device_id = 0 if tensor_to_bind.device.type == "cpu" else tensor_to_bind.device.index,
                element_type = pt_to_np[repr(tensor_to_bind.dtype)],
                shape = tuple(tensor_to_bind.shape),
                buffer_ptr = tensor_to_bind.data_ptr()
            )
        
        return io_binding

    @torch.no_grad()
    def forward(self, output_names: Optional[List[str]] = None, **prepared_inputs: Any) -> Dict[str, Any]:
        """
        Forward function of ORTInferenceModule

        params:
            output_names (Optional[List[str]]): List of output names, in case user doesn't want to bind all outputs of the session
            **prepared_inputs (Any): inputs to model

        returns:
            ort.IOBinding: IOBinding that binds the input buffer and output buffer to the ONNX Runtime Session
        """
        # ensure torch tensors are contiguous, for compatibility with IO Bindings
        prepared_inputs = tree_map_only(torch.Tensor, lambda t: t.contiguous(), prepared_inputs)
        # allocate new memory for output buffer, so there is no overlap with input
        output_buffer = self.output_buffer_creator.create_buffer()
        # Bind IO
        io_binding = self.bind_io(session=self.session, 
                                  onnx_name_to_tensor=self.onnx_name_to_tensor, 
                                  input_buffer=prepared_inputs, 
                                  output_buffer=output_buffer, 
                                  output_names=output_names)

        # binds extra outputs

        self.session.run_with_iobinding(io_binding)

        return output_buffer