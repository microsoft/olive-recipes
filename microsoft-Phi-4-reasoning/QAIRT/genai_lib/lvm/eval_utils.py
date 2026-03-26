#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" common utilities and class implementation for generating test vector, accuracy analysis """

import contextlib
import inspect
import math
import os
import re
import resource
import sys
from typing import Optional, Callable, Dict, Tuple, List, Union, Any, Iterator

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, save

from aimet_common.utils import AimetLogger
from genai_lib.lvm.calibration import ModelDataReader
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

def eval_model(model: torch.nn.Module,
               model_input: Tuple[torch.Tensor],
               intermediate_modules_names: Tuple[str] = None):
    """
    eval models with optionally enabling intermediate tensors.
    :param model: fp32 model or quantsim model.
    :param model_input: prepared model input.
    :param intermediate_modules_names:  list of intermediate modules to dump tensors for.
    """

    intermediate_tensors = {}
    hooks = []
    if intermediate_modules_names:
        modules_to_name = {module: name for name, module in model.named_modules() if name in intermediate_modules_names}

        modules_not_found =  set(intermediate_modules_names) - set(modules_to_name.values())
        if modules_not_found:
            raise ValueError(f'intermediate module name(s)={modules_not_found} not found in model: {type(model)}, '
                             f'defined at {inspect.getfile(model.__class__)}')

        def hook(module: torch.nn.Module, _, output):
            if isinstance(output, torch.Tensor):
                intermediate_tensors[modules_to_name[module]] = np_tensor(output)

        for module in modules_to_name.keys():
            hooks.append(module.register_forward_hook(hook))

    with torch.no_grad():
        if isinstance(model_input, torch.Tensor):
            output = model(model_input)
        else:
            output = model(*model_input)

    for hook in hooks:
        hook.remove()

    return output, intermediate_tensors


def generate_vectors(output_path: str,
                     sim_model: torch.nn.Module,
                     prepared_model_reader: ModelDataReader = None,
                     sample_inputs: Union[List[torch.Tensor], Tuple[torch.Tensor]] = None,
                     format_model_inputs_fn: Iterator[Dict[str, torch.Tensor]] = None,
                     input_names: Optional[List[str]] = None,
                     output_names: Optional[List[str]] = None,
                     align_fp_model: bool = True,
                     num_samples: int = 1,
                     intermediate_modules_names: Tuple[str] = None):
    """
    Generate test vectors and sample inputs for on-target compilation steps.
    :param output_path: Path to save exported DLC file
    :param sim_model: quantsim model to generate quantized o/p for on-target verification.
    :param prepared_model_reader: recorder with calibration data. Mutually exclusive with sample_inputs
    :param sample_inputs: iterable with calibration data. Mutually exclusive with prepared_model_reader
    :param format_model_inputs_fn: user-provided function to adjust model inputs for export
    :param input_names: input names to use for inputs tensors
    :param output_names:  input names to user for output tensors
    :param align_fp_model: if True, extracts fp model from sim user-provided function to adjust model inputs for export
    :param num_samples: number of inputs to use for sampling
    :param intermediate_modules_names:  list of intermediate modules to dump tensors for.
    """
    if (prepared_model_reader is not None) and (sample_inputs is not None): 
        raise ValueError("Expected only one of prepared_model_reader or sample_inputs to be passed, both were passed in this case")
    elif (prepared_model_reader is None) and (sample_inputs is None): 
        raise ValueError("Expected atleast one of prepared_model_reader or sample_inputs to be passed, both were not passed in this case")
    fp_vectors = []
    sim_vectors = []
    prepared_model = QuantizationSimModel.get_original_model(sim_model) if align_fp_model else None
    os.makedirs(output_path, exist_ok=True)
    if input_names is None:
        input_names = list(inspect.signature(sim_model.forward).parameters.keys())

    if format_model_inputs_fn is None:
        def dummy_format(x: Union[Dict, Tuple]) -> Union[Dict, Tuple]:
            yield x
        format_model_inputs_fn = dummy_format

    def dump(tensors: List[np.ndarray], names: List[str] = None) -> Dict:
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        if names:
            assert len(tensors) == len(names), f'Length mismatch: tensors length is {len(tensors)}, name length is {len(names)}.'
        else:
            names = [f't{i}' for i in range(len(tensors))]
        return dict(zip(names, tensors))
    
    def eval_prepared_and_quantsim_models(model_inputs: Tuple):
        
        if prepared_model:
            fp_outputs, intermediate_tensors = eval_model(prepared_model, model_inputs, intermediate_modules_names)
            outputs = dump(np_tensor(fp_outputs), output_names)
            outputs.update(intermediate_tensors)
        _sim_outputs, intermediate_tensors = eval_model(sim_model, model_inputs, intermediate_modules_names)
        sim_outputs = dump(np_tensor(_sim_outputs), output_names)
        sim_outputs.update(intermediate_tensors)
        model_inputs = dump(np_tensor(model_inputs), input_names)

        fp_vectors.append({'inputs': model_inputs, 'outputs': np_tensor(outputs)})
        sim_vectors.append({'inputs': model_inputs, 'outputs': sim_outputs})
        
    with torch.no_grad():

        for i in tqdm(range(min(len(prepared_model_reader if sample_inputs is None else sample_inputs), num_samples))):
            if num_samples is not None and num_samples == i:
                break
            for formatted_input in format_model_inputs_fn(prepared_model_reader.get_flattened_inputs(i) if sample_inputs is None else sample_inputs[i]):
                model_inputs = tuple(formatted_input.values()) if isinstance(formatted_input,dict) else formatted_input
                eval_prepared_and_quantsim_models(model_inputs)

    np.save(f'{output_path}/fp.npy', fp_vectors, allow_pickle=True)
    np.save(f'{output_path}/int8.npy', sim_vectors, allow_pickle=True)

    inputs_dir = os.path.join(output_path, 'raw_inputs')
    os.makedirs(inputs_dir, exist_ok=True)
    sample_inputs = sim_vectors[0]['inputs']
    with open(os.path.join(inputs_dir, 'input_list.txt'), 'w') as f:
        for name, tensor in sample_inputs.items():
            filename = f'{name}.raw'
            tensor.astype(np.float32).tofile(os.path.join(inputs_dir, filename))
            f.write(f"{filename} ")


def get_sqnr_nptensor(
    fp_tensor: np.ndarray, sim_tensor: np.ndarray, eps: float, sample_wise: bool = False
) -> float:
    """
    Computes SQNR

    :param fp_tensor: FP32 np array
    :param sim_tensor: np array with QDQ noise.
    :param eps: the smallest positive value to avoid div-by-zero error.
    :param sample_wise: Specifies whether to calculate SQNR per sample before averaging.
        If set to True, It mitigates the dominance of stronger signal samples in the SQNR calculation.
        If set to False, SQNR is calculated across all elements. Defaults to False for backward compatibility.
    :return: Sigal-to-quantization noise ratio in dB scale.
    """
    if fp_tensor.shape != sim_tensor.shape:
        raise ValueError('Both tensors must have the same shape')

    if sample_wise and fp_tensor.ndim < 2:
        raise ValueError(
            'For sample-wise calculation, both tensors must have at least two dimensions'
        )

    sim_error = fp_tensor - sim_tensor

    if sample_wise:
        # Sample-wise calculation (dims_except_sample_dim)
        axis = tuple(range(fp_tensor.ndim)[1:])
    else:
        axis = None

    exp_noise = (sim_error**2).mean(axis=axis) + eps
    exp_signal = (fp_tensor**2).mean(axis=axis)
    sqnr_db = 10 * (np.log10(exp_signal) - np.log10(exp_noise))
    return sqnr_db.mean()


def get_sqnr(
    fp_tensor: torch.Tensor,
    sim_tensor: torch.Tensor,
    eps: float = sys.float_info.min,
    sample_wise: bool = False,
) -> float:
    """
    Computes SQNR

    :param fp_tensor: FP32 torch tensor
    :param sim_tensor: torch tensor with QDQ noise.
    :param eps: the smallest positive value to avoid div-by-zero error.
    :param sample_wise: Specifies whether to calculate SQNR per sample before averaging.
        If set to True, it mitigates the dominance of stronger signal samples in the SQNR calculation.
        If set to False, SQNR is calculated across all elements. Defaults to False for backward compatibility
    :return: Sigal-to-quantization noise ratio in dB scale.
    """
    return get_sqnr_nptensor(
        np_tensor(fp_tensor), np_tensor(sim_tensor), eps, sample_wise
    )


def np_tensor(tensor):
    """
    :return: numpy tensor(s) from torch tensor(s)
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    if isinstance(tensor, (list, tuple)):
        cls = tuple if isinstance(tensor, tuple) else list
        return cls(np_tensor(x) for x in tensor)

    if isinstance(tensor, dict):
        return {key: np_tensor(value) for key, value in tensor.items()}

    return tensor


def compute_sqnr(model_inputs, fp_model, sim_model, sample_wise: bool = False) -> float:
    """
    method to compute sqnr given model inputs as an iterable, fp and sim models

    :param model_inputs: input to the models, iterable
    :param fp_model: floating point model
    :param sim_model: quantsim model
    :param sample_wise: Specifies whether to calculate SQNR per sample before averaging.
        If set to True, it mitigates the dominance of stronger signal samples in the SQNR calculation.
        If set to False, SQNR is calculated across all elements. Defaults to False for backward compatibility
    """
    fp_outs, qt_outs = [], []
    with torch.no_grad():
        for i, img in enumerate(model_inputs):
            embeddings_fp = fp_model(img)
            fp_outs.append(embeddings_fp)
            embeddings_qt = sim_model(img)
            qt_outs.append(embeddings_qt)
    sqnr = get_sqnr(
        torch.stack(fp_outs).cpu(), torch.stack(qt_outs).cpu(), sample_wise=sample_wise
    )
    return sqnr

    
def analyze_layerwise(sim: Union[QuantizationSimModel, Any],
                      calibration_data: ModelDataReader,
                      prepared_model: torch.nn.Module,
                      use_prev_layer_quant_output: bool = False,
                      num_samples: int = 1,
                      output_path: str = None,
                      match_name: str = None,
                      export_csv_and_plot: bool = False):
    """
    method to dump and log layerwise stats (FP vs quantsim).
    :param sim: quantization simulation model.
    :param calibration_data: calibration data to use for comparing sim against fp
    :param prepared_model: use prepared model as reference for FP.
    :param use_prev_layer_quant_output:  supports two options for layer wise analysis based on the flag
        True => sim input -> fp/sim output (if prepared model not provided use wrapped fp module)
        False => fp input  -> fp/sim output
    :param num_samples: number of inputs to use for sampling
    :param output_path: path to save the stats as pickle file.
    :param match_name: regex string to match one or more layers
    :param export_csv_and_plot: export layerwise stats as csv and plot the bottom 25 sqnr values
    :return: optimized and calibrated quantsim model.
    """
    modules_stats = {}
    fp_to_quant_modules = {}
    quant_to_fp_modules = {}

    def hook(module: torch.nn.Module, inputs, output):
        """ Save model input tuple of tensors and output tensor"""
        kwargs = inspect.currentframe().f_back.f_locals['kwargs']
        if isinstance(module, BaseQuantizationMixin):
            quant_module = module
            fp_module = quant_to_fp_modules[module]
            fp_output, sim_output = fp_module(*inputs, **kwargs), output
        else:
            quant_module = fp_to_quant_modules[module]
            fp_output, sim_output = output, quant_module(*inputs, **kwargs)

        if isinstance(fp_output, (tuple,list)) and isinstance(sim_output, (tuple, list)):
            modules_stats[quant_module]['down-selected'] = len(fp_output),len(sim_output)
            fp_tensor, sim_tensor =np_tensor(fp_output[0]), np_tensor(sim_output[0])
        else:
            fp_tensor, sim_tensor =np_tensor(fp_output), np_tensor(sim_output)

        try:
            modules_stats[quant_module]['sqnr'].append(get_sqnr_nptensor(fp_tensor, sim_tensor, sys.float_info.min))
            modules_stats[quant_module]['min'].append({'fp': np.min(fp_tensor), 'int': np.min(sim_tensor)})
            modules_stats[quant_module]['max'].append({'fp': np.min(fp_tensor), 'int': np.min(sim_tensor)})
        except Exception as e:
            modules_stats[quant_module]['skipped_on_error'] = type(e)

    name_to_fp_modules = {name: module for name, module in prepared_model.named_modules()} if prepared_model else {}

    hooks = []
    for name, quant_module in sim.quant_wrappers():
        modules_stats[quant_module] = {'name': name, 'sqnr': [], 'min': [], 'max': []}
        if not prepared_model:
            fp_module = quant_module._module_to_wrap # pylint: disable = protected-access
        else:
            fp_module = name_to_fp_modules[name]
        fp_to_quant_modules[fp_module] = quant_module
        quant_to_fp_modules[quant_module] = fp_module
        if use_prev_layer_quant_output:
            hooks.append(quant_module.register_forward_hook(hook))
        else:
            hooks.append(fp_module.register_forward_hook(hook))

    with torch.no_grad():
        for i, inputs in enumerate(tqdm(calibration_data, total=(min(len(calibration_data),num_samples)))):
            if i == num_samples:
                break
            _ = prepared_model(*inputs) if not use_prev_layer_quant_output else sim.model(*inputs)

    for hook in hooks:
        hook.remove()

    if match_name:
        match_name = re.compile(match_name)
    logger.info('SQNR layerwise stats (num_layers=%d) for  %s (%s):', len(modules_stats), type(sim.model),
                ("sim input -> fp/sim output" if use_prev_layer_quant_output else "fp input  -> fp/sim output") )
    for i, stats in enumerate(modules_stats.values()):
        mean_sqnr  = stats['mean_sqnr'] = np.array(stats['sqnr']).mean()
        if (match_name is None and not math.isnan(mean_sqnr) and mean_sqnr < 100) \
                or (match_name and re.match(match_name, stats['name'])):
            logger.info('%-4.0f\t%-5.2f dB : %s', i, stats['mean_sqnr'], stats['name'])

    if output_path:
        filepath = os.path.join(output_path, 'layerwise_stats' + ('_sim_in' if use_prev_layer_quant_output else '_fp_in'))
        np.save(filepath, list(modules_stats.values()), allow_pickle=True)
        logger.info('saved stats at:%s.npy', filepath)

    if export_csv_and_plot:
        layerwise_stats_df=pd.DataFrame(list(modules_stats.values()))
        layerwise_stats_df=layerwise_stats_df.sort_values('mean_sqnr')
        source = ColumnDataSource(layerwise_stats_df[['name','mean_sqnr']].head(25))
        plot = figure(y_range=layerwise_stats_df['name'].head(25), height=350,width=700,title="Layerwise mean SQNR Value - Bottom 25")
        plot.hbar(y='name', right='mean_sqnr', height=0.9,source=source)
        hover = HoverTool()
        hover.tooltips = [("Name", "@name"), ("Mean SQNR", "@mean_sqnr")]
        plot.add_tools(hover)
        output_file(f'{output_path}/plot.html', mode='inline')
        save(plot)
        layerwise_stats_df=layerwise_stats_df[['name','min','max','mean_sqnr']]
        layerwise_stats_df.to_csv(f"{output_path}/layerwise_data.csv",index=False)

@contextlib.contextmanager
def increase_recursion_limit():
    """
    Utility to temporarily increase recursion limit.
    :return: None
    """
    original_recursion_limit = sys.getrecursionlimit()
    original_rlimit = resource.getrlimit(resource.RLIMIT_STACK)
    try:
        sys.setrecursionlimit(10 ** 9)
        resource.setrlimit(resource.RLIMIT_STACK, (-1, -1))
        yield
    finally:
        sys.setrecursionlimit(original_recursion_limit)
        resource.setrlimit(resource.RLIMIT_STACK, original_rlimit)
