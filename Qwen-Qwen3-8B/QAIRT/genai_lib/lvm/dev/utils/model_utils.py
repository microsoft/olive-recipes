#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" common utilities and class implementation for model architecture adaptation """
from typing import Dict, List, Tuple, Union, Any, Callable

import torch

from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

class ModelOutputIndexFilter(torch.nn.Module):
    """
    Wrap a model that generates several outputs, and filter it to return only a subset of those
    Used to simplify an ONNX model, shrinking its size and preventing generation of unused and unwanted outputs
    """
    def __init__(self, model: torch.nn.Module,
                 output_index_filter: List,
                 forward_kwargs: Dict[str, Any] = None,
                 out_mapper_fn: Callable = None):
        """
        :param model: model whose output is to be filtered
        :param output_index_filter: list matching length of model's outputs, indicating which idx of each output to keep,
                                    None means entirely skipping an output, integers mean selecting output at that index,
                                    a string with a single colon ":" means selecting all entries for the current output
        :param forward_kwargs: additional kwargs to be passed to the model during inference
        :param out_mapper_fn: function to map the filtered outputs to the desired format
        """
        super().__init__()
        self.model = model
        self.output_index_filter = output_index_filter
        self.forward_kwargs = forward_kwargs if forward_kwargs else {}
        self.out_mapper_fn = out_mapper_fn if out_mapper_fn is not None else lambda x: x
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.config = model.config if hasattr(model, "config") else None


    def forward(self, *args, **kwargs):
        """
        :param args: args to be forwarded to the model
        :param kwargs: kwargs to be forwarded to the model
        """
        kwargs.update(self.forward_kwargs)
        outputs = self.model(*args, **kwargs)
        assert len(outputs) == len(self.output_index_filter), f'output_index_filter must match expected model output length, got {len(outputs)} model outputs, but {len(self.output_index_filter)} index filters'
        filtered_outputs = []
        for idx, output_filter in enumerate(self.output_index_filter):
            if output_filter is None:
                continue
            if output_filter == ':':
                chosen_output = outputs[idx]
            else:
                chosen_output = outputs[idx][output_filter]
                # unsqueeze to make up for the lost dimension when we access at [output_filter]
                if isinstance(outputs[idx], torch.Tensor):
                    chosen_output = chosen_output.unsqueeze(0)
                else:
                    chosen_output = (chosen_output, )
            filtered_outputs.append(chosen_output)
        return self.out_mapper_fn(filtered_outputs)
