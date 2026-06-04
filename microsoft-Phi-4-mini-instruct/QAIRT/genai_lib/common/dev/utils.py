# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
# =============================================================================
# Copyright 2020 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
""" Common utilities for GenAI Lib """

import functools
import inspect
import logging
from importlib import metadata, util
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch

from packaging import version

AIMET_TORCH_AVAILABLE = util.find_spec("aimet_torch") is not None


@functools.cache
def is_package_greater_or_equal(package_name: str, package_version: str) -> bool:
    return version.parse(metadata.version(package_name)) >= version.parse(
        package_version
    )


def is_transformers_greater_or_equal(package_version: str) -> bool:
    return is_package_greater_or_equal('transformers', package_version)


is_transformers_greater_or_equal_than_4_48 = is_transformers_greater_or_equal('4.48.0')
is_transformers_greater_or_equal_than_4_51 = is_transformers_greater_or_equal('4.51.0')
is_transformers_greater_or_equal_than_4_53 = is_transformers_greater_or_equal('4.53.0')


def get_aimet_version():
    # Here we are fetching the AIMET version by giving precedence to the open source version (if exists)
    # This function returns the o/p of type packaging.version.Version, and not str.
    package_names = ["AimetTorch", "aimet-torch"]
    aimet_version = None
    for package in package_names:
        try:
            aimet_version = version.parse(metadata.version(package))
        except:
            continue
    assert aimet_version is not None, "Could not find any AIMET version"
    return aimet_version

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def change_signature_defaults(func: Callable, defaults_dict: Dict[str, Any]) -> Callable:
    """
    Utility that changes default values on a function's signature.
    This is useful for boolean inputs that cannot be part of a dummy input for preparation
    as they disappear on the traced graph.

    Args:
        func (Callable): The function whose signature defaults are to be changed.
        defaults_dict (Dict[str, Any]): A dictionary where keys are parameter names and values are the new default values.

    Returns:
        Callable: A new function with the updated signature defaults.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    for i, param in enumerate(params):
        if param.name in defaults_dict:
            params[i] = param.replace(default=defaults_dict[param.name])

    new_sig = sig.replace(parameters=params)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = new_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return func(*bound_args.args, **bound_args.kwargs)

    wrapper.__signature__ = new_sig
    return wrapper


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once


@functools.lru_cache(None)
def info_once(self, *args, **kwargs):
    """
    This method is identical to `logger.info()`, but will emit the info with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.info(*args, **kwargs)


logging.Logger.info_once = info_once


def filter_outputs(outputs: List[Union[torch.Tensor, Tuple]],
                   output_index_filter: List[Optional[Union[int, str]]]) -> Tuple:
    """
    Filters the outputs based on the provided index filter.

    Args:
        outputs (List[Union[torch.Tensor, Tuple]]): The list of model outputs.
        output_index_filter (List[Optional[Union[int, str]]]): The list of index filters.
            Each element can be an integer, a string (':'), or None.
            If integer, that index is kept.
            If string ":", all indexes are kept.
            If None, the output is removed.

    Returns:
        Tuple: A tuple of filtered outputs.

    Raises:
        AssertionError: If the length of outputs does not match the length of output_index_filter.
    """
    assert len(outputs) == len(output_index_filter), \
        f'output_index_filter must match expected model output length, ' \
        f'got {len(outputs)} model outputs, but {len(output_index_filter)} index filters'

    filtered_outputs = []
    for idx, output_filter in enumerate(output_index_filter):
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

    return tuple(filtered_outputs)

if AIMET_TORCH_AVAILABLE:
    from aimet_torch.v2.nn import QuantizationMixin
    from aimet_torch.v2.quantsim import QuantizationSimModel
    from aimet_torch.v2.utils import _ContextManager

    def extract_qmodules(sim, check_type) -> Generator:
        """
        Extracts and returns a generator of qmodules from the given simulation object that are instances of the specified type.

        :param sim: QuantizationSimModel instance.
        :param check_type: The type to check each qmodule against.
        :return: A generator of qmodules that are instances of check_type.
        """
        return (x for x in sim.qmodules() if isinstance(x, check_type))

    def reset_kernels(sim: QuantizationSimModel) -> _ContextManager:
        """
        Resets the kernels of all quantization modules within a QuantizationSimModel to None.

        :param sim: QuantizationSimModel instance.
        :return: A context manager that handles the registration and restoration of kernels.
        """
        orig = {
            qmodule: qmodule.get_kernel()
            for qmodule in extract_qmodules(sim, QuantizationMixin)
        }

        def restore_kernels():
            for qmodule, kernel in orig.items():
                qmodule.set_kernel(kernel)

        ctx = _ContextManager(action=lambda: None, cleanup=restore_kernels)

        try:
            for qmodule in extract_qmodules(sim, QuantizationMixin):
                qmodule.set_kernel(None)
        except Exception:
            ctx._cleanup()
            raise
        else:
            return ctx
