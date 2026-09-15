#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" common utilities and class implementation for calibration """

import contextlib
import glob
import inspect
import os
import pickle
from typing import Tuple, Any, Dict, Optional, List, Union, Callable
from packaging import version

import numpy
import torch
from aimet_torch.adaround import adaround_optimizer
import logging
logger = logging.getLogger(__name__)

FILENAME_PREFIX = 'cal_'
FWD_SIGNATURE_FILE = 'fwd_signature.pkl'
PREP_FWD_MAP_FILE = 'prepared_fwd_map.pkl'

def _flatten_model_inputs(inputs: Dict, prefix: str = '', input_dict: Dict = None, tensors_only: bool = True):
    """ recursive function to flatten flattens a dict or nested dict into flat dict """

    if isinstance(inputs, dict):
        for name, t in inputs.items():
            new_prefix = name if prefix == '' else f'{prefix}_{name}'
            _flatten_model_inputs(t, new_prefix, input_dict, tensors_only)

    elif tensors_only is False or isinstance(inputs, torch.Tensor):
        input_dict[prefix] = inputs

def flatten_model_inputs(inputs: Dict, tensors_only: bool = True) -> Dict[str, Any]:
    """
    function to flatten flattens a dict or nested dict into flat dict
    :param inputs: dict of inputs
    :param tensors_only: if True, filters out entry which do not have Tensor as value
    :return: flat dict.
    """
    input_dict = {}
    _flatten_model_inputs(inputs, '', input_dict, tensors_only)
    return input_dict


# TODO split reader into data_loader (Calibration) and i/o data reader
class ModelDataReader:
    """ Implements a reader for recorded calibration data. """

    def __init__(self, saved_path: str, input_only: bool, is_prepared_model: bool, num_samples: int):
        self.saved_path = saved_path
        self._len = len(glob.glob1(self.saved_path, FILENAME_PREFIX + "*.npy"))
        if num_samples > 0:
            self._len = min(self._len, num_samples)
        assert self._len > 0, f'No sample model inputs found in {self.saved_path}'
        self._input_only = input_only
        self._is_prepared_model = is_prepared_model
        with open(os.path.join(self.saved_path, FWD_SIGNATURE_FILE), 'rb') as f:
            self.model_parameter_defaults = pickle.load(f)
        if is_prepared_model:
            prepared_fwd_path = os.path.join(saved_path, PREP_FWD_MAP_FILE)
            assert os.path.exists(prepared_fwd_path), f'{prepared_fwd_path} does not exist, did you remember to call `recorder.extract_prepare_model_info(prepared_model)` ?'
            with open(os.path.join(self.saved_path, PREP_FWD_MAP_FILE), 'rb') as f:
                self.prep_model_input_map = pickle.load(f)

    def get_calibration_data(self, item: int):
        """Returns the indexed item from the dataset"""
        file_path = f'{self.saved_path}/cal_{item}.npy'
        if item < self._len:
            return numpy.load(file_path, allow_pickle=True)[0]

        raise IndexError

    def __getitem__(self, item: int):
        """Returns the indexed item from the dataset formatted based on config """
        data = self.get_calibration_data(item)

        if self._is_prepared_model:
            recoded_flatten_inputs = self.get_flattened_inputs(inputs=data['inputs'])
            inputs = tuple([recoded_flatten_inputs[input_key] for input_key in self.prep_model_input_map])
        else:
            inputs = self.format_inputs(data['inputs'], self.model_parameter_defaults)

        if self._input_only:
            return inputs

        return inputs, data['outputs']

    def get_flattened_inputs(self, item: int = None, inputs: Optional[Tuple[Any,]] = None, tensor_only: bool = True):
        """Returns the indexed item from the dataset as flattened dict"""
        if inputs is None:
            inputs = self.get_calibration_data(item)['inputs']
        inputs = self.format_inputs(inputs, self.model_parameter_defaults)
        return flatten_model_inputs(dict(zip(self.model_parameter_defaults.keys(), inputs)), tensors_only=tensor_only)

    @staticmethod
    def format_inputs(inputs: Tuple[Tuple, Dict], parameter_defaults):
        """
        flattens input in the format of '*args, **kwargs' to tuple along with adding defaults when value not provided.
        :param inputs:  input tuple in *args, **kwargs format
        :param parameter_defaults: forward pass parameter /w default values
        :return: input as tuple.
        """
        *positional_inputs, kwargs = inputs
        if kwargs:
            add_args = []
            param_keys = list(parameter_defaults.keys())
            if param_keys[0] == "self":
                param_keys = param_keys[1:]
            for k in param_keys[len(positional_inputs):]:
                if k in kwargs:
                    add_args.append(kwargs[k])
                else:
                    default = parameter_defaults[k]
                    if default != inspect.Parameter.empty:
                        add_args.append(default)
                    else:
                        ValueError(f'no value provided for kwarg={k}, which has no defaults')
            inputs = tuple([*positional_inputs, *add_args])
        else:
            inputs = tuple(positional_inputs)
        return inputs

    def __len__(self):
        """Returns length of calibration data"""
        return self._len


class ModelDataRecorder:
    """
    Implements mechanism to capture model input and output as calibration data.
    """
    def __init__(self, save_path: str, model: Optional[torch.nn.Module] = None,
                 clean_start: bool = False, num_samples: int = -1):
        self.index = 0
        self.hook_handler = []
        self.input_args_type = None
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.num_samples = num_samples

        if model is not None:
            self._register(model)

            calibration_files = glob.glob(f'{self.save_path}/{FILENAME_PREFIX}*.npy')
            if len(calibration_files) > 0:
                logger.info("calibration data (num=%d) found at %s, %s ..",
                            len(calibration_files), self.save_path, 'deleting' if clean_start else 'exiting')
                if clean_start:
                    for filename in calibration_files:
                        os.remove(filename)
                else:
                    raise ValueError("if you intend to collect calibration data again, please set clean_start=True")
        else:
            fwd_signature_path = os.path.join(self.save_path, FWD_SIGNATURE_FILE)
            assert os.path.exists(fwd_signature_path), f'forward signature file not found in {fwd_signature_path}'

    def _register(self, model: torch.nn.Module):
        """
        Register forward hook for model input/output.
        """
        with open(os.path.join(self.save_path, FWD_SIGNATURE_FILE), 'wb') as f:
            params_defaults = {param: value.default for param, value in
                               inspect.signature(model.forward).parameters.items()}
            pickle.dump(params_defaults, f, protocol=pickle.HIGHEST_PROTOCOL)

        def hook(_, inputs, outputs):
            """ Save model input tuple of tensors and output tensor"""

            fn_context = [f.function for f in inspect.stack()[1:]] # skipping innermost stack-frame[0] i.e. hook(..)
            if version.parse(torch.__version__) < version.parse("2.5"):
                assert fn_context[0] == '_call_impl',\
                    f'unexpected context={fn_context}, hook expected to be invoked from torch.nn.module._call_impl(..) proxy fn'
            else:
                assert fn_context[:2] == ['inner', '_call_impl'], \
                    f'unexpected context={fn_context} , hook expected to be invoked from torch.nn.module._call_impl.inner(..) proxy fn'

            forward_fn = inspect.currentframe().f_back
            inputs = *inputs, forward_fn.f_locals['kwargs']
            self._verify_model_inputs(inputs)
            numpy.save(
                f'{self.save_path}/{FILENAME_PREFIX}{self.index}',
                [{'inputs' : inputs, 'outputs': outputs}], allow_pickle=True)
            self.index += 1

            if self.index == self.num_samples:
                self.stop_recording()

        self.hook_handler.append(model.register_forward_hook(hook))

    def _verify_model_inputs(self, inputs):
        """
        Implements a check to ensure the inputs args are not changing between model invocation.
        :param inputs: inputs captured with current invocation of hook
        """
        *positional_inputs, kwargs = inputs
        input_args_type = [type(x) for x in positional_inputs] + [(n,type(v)) for n,v in kwargs.items()]
        if self.input_args_type is None:
            self.input_args_type = input_args_type
        else:
            assert self.input_args_type == input_args_type, f'unsupported option, inputs arguments are changing, expected `{self.input_args_type}`, got `{input_args_type}`'

    def stop_recording(self):
        """ helper method to remove all registered hook in the model being observed. """
        if self.hook_handler:
            for hook in self.hook_handler:
                hook.remove()
            self.hook_handler = []
            logger.info("removed hooks for model, calibration data (num=%d) saved at:%s",
                        self.index, self.save_path)

    def extract_prepare_model_info(self, model: torch.nn.Module) -> List[str]:
        """
        Register forward hook for model input/output.
        """

        param_keys = inspect.signature(model.forward).parameters.keys()
        file_path = f'{self.save_path}/cal_0.npy'
        assert os.path.isfile(file_path), f'{file_path} does not exist, did you observe data flowing through the model with this recorder?'
        inputs = numpy.load(file_path, allow_pickle=True)[0]['inputs']
        with open(os.path.join(self.save_path, FWD_SIGNATURE_FILE), 'rb') as f:
            model_parameter_defaults = pickle.load(f)
        inputs = ModelDataReader.format_inputs(inputs, model_parameter_defaults)
        input_keys = flatten_model_inputs(dict(zip(model_parameter_defaults.keys(), inputs)), tensors_only=True).keys()
        assert len(param_keys) == len(input_keys), f'Prepared model forward expected {len(param_keys)} inputs=`{param_keys}`, \
                                                     but observed calibration data contains {len(input_keys)} inputs=`{input_keys}`'

        with open(os.path.join(self.save_path, PREP_FWD_MAP_FILE), 'wb') as f:
            input_param_amp = dict((zip(input_keys, param_keys)))
            logger.info("calibration inputs to args mapping: %s", input_param_amp)
            pickle.dump(input_param_amp, f, protocol=pickle.HIGHEST_PROTOCOL)
        return list(param_keys)


    @property
    def reader(self) -> ModelDataReader:
        """ return an input and output data reader """
        return ModelDataReader(self.save_path, input_only=False, is_prepared_model=False, num_samples=self.num_samples)

    @property
    def calibration_data(self) -> ModelDataReader:
        """ return an input data reader """
        return ModelDataReader(self.save_path, input_only=True, is_prepared_model=False, num_samples=self.num_samples)

    @property
    def prepared_model_calibration_data(self) -> ModelDataReader:
        """ return an input data reader """
        return ModelDataReader(self.save_path, input_only=True, is_prepared_model=True, num_samples=self.num_samples)

    @property
    def prepared_model_reader(self) -> ModelDataReader:
        """ return an input data reader """
        return ModelDataReader(self.save_path, input_only=False, is_prepared_model=True, num_samples=self.num_samples)


def get_output_names(outputs: Any) -> Optional[List[str]]:
    """
    provides default output names for LVM models
    :param outputs: FP32 model output
    :return: return list of output names or None
    """
    if hasattr(outputs, 'to_tuple'):
        outputs = outputs.to_tuple()
    if isinstance(outputs, tuple) and isinstance(outputs[0], torch.Tensor):
        return [f'output{i}' for i in range(len(outputs))]
    elif isinstance(outputs, torch.Tensor):
        return ['output']
    return None

def nested_map(data, fn: Callable[[torch.Tensor], torch.Tensor]):
    """
    Apply a function to a nested tuple, list, or dict of tensors.
    :param data: Tensor, or a nested tuple, list, or dict of tensors.
    :param fn: Function to apply to the tensors
    :return: Nested structure of tensors with function applied
    """
    if isinstance(data, torch.Tensor):
        return fn(data)

    if isinstance(data, (tuple, list)):
        cls = tuple if isinstance(data, tuple) else list
        return cls(nested_map(x, fn) for x in data)

    if isinstance(data, dict):
        return {
            key: nested_map(value, fn) for key, value in data.items()
        }

    logger.debug('unexpected input type=%s, expecting torch.Tensor, tuple, list, or dict. skipping..', type(data))
    return data

def change_tensor_dtype(data: Union[torch.Tensor, List, Tuple, Dict], dtype: torch.dtype):
    """
    Change the data's dtype.
    :param data: Tensor, or a nested tuple, list, or dict of tensors.
    :param dtype: dtype
    :return: data with modified dtype.
    """
    return nested_map(data, lambda x: x.to(dtype=dtype) if x.is_floating_point() else x)


@contextlib.contextmanager
def get_num_adaround_iterations(dtype: torch.dtype, num_iterations: int):
    """
    Sets adaround config for fp16 execution from default fp32.
    """
    adaround_batch_size = adaround_optimizer.BATCH_SIZE
    if dtype == torch.half:
        adaround_optimizer.BATCH_SIZE = adaround_batch_size * 2
        num_iterations = int(num_iterations / 2)
    yield num_iterations
    adaround_optimizer.BATCH_SIZE = adaround_batch_size
