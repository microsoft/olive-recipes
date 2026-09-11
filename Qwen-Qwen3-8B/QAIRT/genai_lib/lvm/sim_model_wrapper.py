#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Proxy model to insert sim.model into an existing model workflow """
import inspect
from typing import Callable, Optional

import torch
from diffusers import DiffusionPipeline

from aimet_common.utils import AimetLogger
from genai_lib.lvm.calibration import flatten_model_inputs, ModelDataReader
from aimet_torch.utils import get_device

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class ModelWrapper(torch.nn.Module):
    """
    Adapter class to enable interfacing sim object within original model context.
    """
    def __init__(self,
                 model_to_wrap: torch.nn.Module,
                 allow_passthrough: bool,
                 model: torch.nn.Module,
                 output_mapper_function: Callable):
        super(ModelWrapper, self).__init__()
        self.original_forward_parameters = inspect.signature(model_to_wrap.forward).parameters
        self._model = model
        self.pruned = False
        self.config = getattr(model_to_wrap, 'config', None)
        self._model_to_wrap = model_to_wrap if allow_passthrough else None
        self._output_mapper_fn = output_mapper_function

    def forward(self, *inputs, **kwargs):
        """ forward proxy """
        inputs = ModelDataReader.format_inputs((*inputs, kwargs),  self.original_forward_parameters)
        inputs = tuple(flatten_model_inputs(dict(zip(self.original_forward_parameters.keys(), inputs))).values())
        outputs = self._modules['_model'](*inputs)
        outputs = self._output_mapper_fn(outputs)
        return outputs

    def __getattr__(self, name):
        """
        method to allow forwarding getattr request to the original model
        """
        try:
            model = self._modules['_model']
            if name == 'device':
                return get_device(model)
            if name == 'dtype':
                return next(model.parameters()).dtype

            return self.__dict__[name]
        except KeyError as e:
            info = inspect.getframeinfo(inspect.currentframe().f_back)
            original_model = self._modules['_model_to_wrap']
            if original_model is None:
                logger.error('Unsupported access for %s on prepared model (%s), from %s (%s #%s), ',
                             name, type(model), info.function, info.filename, info.lineno)
                raise e
            item = getattr(original_model, name)
            if isinstance(item, torch.nn.Module):
                logger.error('Unsupported access for %s of prepared model modules (%s), from %s (%s #%s), ',
                             name, type(original_model), info.function, info.filename, info.lineno)
            return item


def set_model_in_pipeline(model,
                          target: DiffusionPipeline,
                          target_module: str,
                          use_wrapper: bool,
                          output_mapper_function: Optional[Callable] = None,
                          allow_passthrough: bool = False):
    """
    replaces the target module with provided model instance
    :param model: model to replace with.
    :param target: pipeline containing the target module
    :param use_wrapper: If True, sets Wrapper model to adapt with original model context.
    :param target_module: instance path with the target module e.g. 'vae.decoder'
    :param output_mapper_function: maps the prepared model output which is tensor or Tuple of Tensor
    :param allow_passthrough: if True, allow accessing to wrapped model for attribute access.
    """
    if '.' in target_module:
        *parent, target_module = target_module.split('.')
        for m in parent:
            target = getattr(target, m)

    if not use_wrapper:
        setattr(target, target_module, model)
        return

    if output_mapper_function is None:
        output_mapper_function = lambda x: x

    model_to_wrap = getattr(target, target_module)
    logger.info('wrapping model(%s) @ %s', type(model_to_wrap), target_module)
    setattr(target, target_module, ModelWrapper(model_to_wrap, allow_passthrough, model, output_mapper_function))
