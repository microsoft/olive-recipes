#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""OSET utilities for GenAI Lib"""

from typing import Iterable, Optional, Type, TypeAlias

import torch
from aimet_common.utils import AimetLogger
from aimet_torch.v2 import nn
from aimet_torch.v2.nn.modules import custom as custom_ops
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.utils import _ContextManager

from genai_lib.common.dev.utils import extract_qmodules

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

try:
    from aimet.aimet_oset import OsetLib
except ImportError:
    _logger.error(
        'In order to import OsetLib, the OSET repository must be added to module search path such as PYTHONPATH'
    )
    raise
else:
    _ModuleType: TypeAlias = Type[torch.nn.Module]
    _QMODULE_TYPE_TO_KERNEL = {
        # Linear, Conv, MatMul ops
        nn.QuantizedLinear: OsetLib.linear,
        nn.QuantizedConv2d: OsetLib.conv2d,
        nn.QuantizedConvTranspose2d: OsetLib.conv_transpose2d,
        custom_ops.QuantizedMatMul: OsetLib.matmul,
        # Norm ops
        nn.QuantizedLayerNorm: OsetLib.layer_norm,
        nn.QuantizedInstanceNorm2d: OsetLib.instance_norm,
        nn.QuantizedGroupNorm: OsetLib.group_norm,
        # Elementwise ops
        custom_ops.QuantizedAdd: OsetLib.add,
        custom_ops.QuantizedSubtract: OsetLib.sub,
        custom_ops.QuantizedMultiply: OsetLib.mul,
        custom_ops.QuantizedDivide: OsetLib.div,
        custom_ops.QuantizedConcat: OsetLib.cat,
        custom_ops.QuantizedSin: OsetLib.sin,
        custom_ops.QuantizedCos: OsetLib.cos,
        # Activation ops
        nn.QuantizedGELU: OsetLib.gelu,
        nn.QuantizedSigmoid: OsetLib.sigmoid,
        nn.QuantizedSoftmax: OsetLib.softmax,
    }

    def register_oset_kernels(
        sim: QuantizationSimModel,
        module_types_to_exclude: Optional[Iterable[_ModuleType]] = None,
        modules_to_exclude: Optional[Iterable[torch.nn.Module]] = None,
    ) -> _ContextManager:
        """
        Registers OSET kernels for quantization modules within a QuantizationSimModel, with options to exclude certain types or instances of modules.
        Ensures original kernels can be restored after registration with context manager block.

        :param sim: QuantizationSimModel instance.
        :param module_types_to_exclude: Optional iterable of module types to exclude from kernel registration.
        :param modules_to_exclude: Optional iterable of specific module instances to exclude from kernel registration.
        :return: A context manager that handles the registration and restoration of kernels.
        """
        if module_types_to_exclude is None:
            module_types_to_exclude = tuple()
        else:
            module_types_to_exclude = tuple(module_types_to_exclude)

        if modules_to_exclude is None:
            modules_to_exclude = set()
        else:
            modules_to_exclude = set(modules_to_exclude)

        def register_kernel_if_applicable():
            for qmodule in extract_qmodules(sim, nn.QuantizationMixin):
                if isinstance(qmodule, module_types_to_exclude):
                    continue

                if qmodule in modules_to_exclude:
                    continue

                kernel = _QMODULE_TYPE_TO_KERNEL.get(type(qmodule))
                if kernel is not None:
                    qmodule.set_kernel(kernel)
                    _logger.info_once(
                        f'{type(qmodule)} will be executed with the OSET kernel'
                    )
                else:
                    _logger.warning_once(
                        f'There is no OSET kernel corresponding to {type(qmodule)}. It will be executed with the PyTorch kernel'
                    )

        orig = {
            qmodule: qmodule.get_kernel()
            for qmodule in extract_qmodules(sim, nn.QuantizationMixin)
        }

        def restore_kernels():
            for qmodule, kernel in orig.items():
                qmodule.set_kernel(kernel)

        ctx = _ContextManager(action=lambda: None, cleanup=restore_kernels)

        try:
            register_kernel_if_applicable()
        except Exception:
            ctx._cleanup()
            raise
        else:
            return ctx
