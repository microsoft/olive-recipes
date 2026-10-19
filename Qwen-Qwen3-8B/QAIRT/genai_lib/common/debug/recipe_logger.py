#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import logging
from datetime import datetime
import os
import torch
import types
import inspect
from importlib.metadata import version as impLib_version
from packaging import version
from enum import Enum
filter_list = (torch.utils.data.dataloader.DataLoader, torch.nn.Module, torch.Tensor)
logger = None
json_dump = {'Environment': [], 'Property':[], 'Metric': [], "Quantization_PTQ_API":[]}
log_json_path = None
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            if isinstance(obj, types.FunctionType):
                return "{Function}:" + obj.__name__
            if obj.__class__.__name__ == "SeqMseParams":
                return obj.__dict__
            return repr(obj)

class RecipeLogger:
    '''
    Defines a singleton class for initializing the logger and avoiding re-initializing the logger.
    '''
    _instance = None

    def __new__(cls, output_dir, log_path_suffix=None):
        if cls._instance is None:
            cls._instance = super(RecipeLogger, cls).__new__(cls)
            cls._instance._initialize(output_dir, log_path_suffix)
        return cls._instance

    def _initialize(self, output_dir, log_path_suffix=None):
        global logger, log_json_path
        if log_path_suffix:
            log_file_path_prefix = os.path.join(output_dir, "_" + str(log_path_suffix))
        else:
            log_file_path_prefix = os.path.join(output_dir, "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        log_file_path = log_file_path_prefix + ".log"
        log_json_path = log_file_path_prefix + ".json"
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n %(message)s')
        logger = logging.getLogger("LLMDebug")
        logger.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler(log_file_path)
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.info("Logger initialized \n")

def recipe_dump_init(output_dir, log_path_suffix=None):
    """
    This function is responsible for setting up the logger to log the recipe into the specified output directory.
    :param output_dir: output dir to save the output log to
    :param log_path_suffix: optional suffix for the log file name
    """
    return RecipeLogger(output_dir, log_path_suffix)


class Property(Enum):
    """
    Defines the properties for the logging
    dataset_name: The name of the dataset used
    dataset_split: usually refers to "train" or "test" or "validation"
    batch_size: number of samples in a single batch
    num_batches: number of bathces used for a technique
    mask_neg: masking value for the causal attention mask
    ARN : maximum number of tokens that can be consumed by the model at each inference
    context_len: maximum number of tokens that the model can consume in total
    model_precision: represents whether the model was run in float16 or float32 precision
    model_config: model config dictionary
    model_id_or_path: path to model weights or the model_id
    tokens_per_sample:  number of tokens per sample can vary.
    Currently, we set the block_size during dataset preprocessing to match the context length.
    However, each sample might be shorter or longer than the model’s context length (Long Context)
    """

    dataset_name = "dataset_name"
    dataset_split = "dataset_split"
    batch_size = "batch_size"
    num_batches = "num_batches"
    mask_neg = "mask_neg"
    context_length = "context_length"
    model_precision = "model_precision"
    model_config = "model_config"
    model_id_or_path = "model_id_or_path"
    ARN = "ARN"
    tokens_per_sample = "tokens_per_sample"


class Metric(Enum):
    """ Defines the various metrics we can log """
    ppl = "ppl"
    mmlu = "mmlu"
    sqnr = "sqnr"


class ModelType(Enum):
    """ Defines the model type we can log the metric/ property against"""
    hf_model = "hf_model"
    prepared_model = "prepared_model"
    qsim_model = "qsim_model"
    adapted_model = "adapted_model"


def llm_lib_log_env_info(additional_env_info = None):
    """
    This function logs environment information, such as the versions of `aimet_torch` and `transformers`.
    It should ideally be invoked after the logger is initially set up.

    :param additional_env_info: a dictionary which may take in the additional environment information which a
    user wants to log in addition to the transformers, torch and aimet version
    """
    env_var_list = ['AimetTorch', 'transformers', 'torch', 'aimet-torch']
    found_vars={}
    for env_var in env_var_list:
        try:
            env_version = version.parse(impLib_version(env_var))
            found_vars[env_var] = env_version
        except:
            logger.warning(f"package {env_var} not found in the environment variables")

    if additional_env_info is not None:
        for env_var, env_version in additional_env_info.items():
            found_vars[env_var] = env_version

    log_message = "Environment variables found:\n" + "\n".join([f"{key}: {value}" for key, value in found_vars.items()])
    logger.info(log_message)
    json_dump['Environment'].append(found_vars)


def llm_lib_log_and_execute(target_func, additional_properties=dict()):
    """
    This function creates a wrapper around the func, and logs the arguments, kwargs and additional properties before invoking func.
    Since the arguments might include models or input tensors that aren't useful for logging, we filter them out before logging.

    :param
    target_func: the function whose args and kwargs we want to log
    additional_properties: an optional dictionary which contains the keys from the pre-defined Property Enum.
    """
    def wrapper(*args, **kwargs):
       bound_args = inspect.signature(target_func).bind(*args, **kwargs)
       bound_args.apply_defaults()
       args_ = _filter_collection(dict(bound_args.arguments))
       for k, v in additional_properties.items():
            if k not in Property:
                raise KeyError("Property {} not defined".format(k))

       property_payload = {str(k): v for k, v in additional_properties.items()}
       payload = {"Module_name": target_func.__module__, "Operation_name": target_func.__name__, "Parameters_of_algo": args_, "Additional_properties": property_payload}
       json_dump['Quantization_PTQ_API'].append(payload)
       logger.info(f"Module_name : {target_func.__module__} \n Operation_name: {target_func.__name__} \n Parameters_of_algo: {args_}, \n Additional properties: {additional_properties} \n")
       return target_func(*args, **kwargs)

    return wrapper


def llm_lib_log_property(property_dict = None):
    """
    This function is responsible for logging the property if it exists in the Property Enum

    params:
    property_dict: a dictionary containing the keys from one of the pre-defined property Enum with it's corresponding value.
    """
    payload = {}
    if property_dict:
        for property_name, value in property_dict.items():
            if property_name not in Property:
                raise KeyError("given Property is not supported")

            log_message = f"{property_name.value} : {value} \n"
            payload[property_name.value] = value

        logger.info(log_message)
        json_dump['Property'].append(payload)


def llm_lib_log_metric(model_type, metric_name, value, model_name=None):
    """
    This function is responsible for logging the property if it exists in the Metric Enum

    params:
    model_type: a free form user passed model type, hf_model, QSim_model
    metric_name: property passed which comes from the predefined Metric Enum
    value: value corresponding to the Metric
    model_name: name of the model whose property is being logged. Useful to differentiate models with different adapters in Lora.
    """
    if metric_name in Metric and model_type in ModelType:
        metric_payload = {"Model_type": model_type.value, "Metric_name": metric_name.value, "Value": value}

        if model_name is not None:
            metric_payload["Model_name"]=model_name

        log_message = "\n".join([f"{key}: {value}" for key, value in metric_payload.items()])
        logger.info(log_message)
        json_dump['Metric'].append(metric_payload)
    else:
        raise KeyError("given Metric or ModelType is not supported")


def dump_logs_to_json():
    '''
    This function dumps the logs into a JSON file.
    '''
    with open(log_json_path, "w") as json_dump_file:
        json.dump(json_dump, json_dump_file, indent=4,cls=CustomEncoder)
    return log_json_path

def _filter_collection(args):
    """
    This function filters the arguments by removing any noisy or verbose types specified in the filter_list,
     or if the value corresponding to some key contains the object of QuantizationSimModel or it removes the
      iterable (list) if it contains elements from the filter list. Additionally, to keep the logs clean,
      we also remove the nested empty list/ tuple for instance arising from the dummy input.
    :param args: function args or kwargs passed.
    """
    if isinstance(args, dict):
        filtered_dict = {}
        for k, v in args.items():
            if not (isinstance(v, filter_list) or v.__class__.__name__ == 'QuantizationSimModel'):
                filtered_value = _filter_collection(v)
                # filter out if the filtered_value is either an empty list or tuple, for instance in dummy input
                if filtered_value not in ([], ()):
                    filtered_dict[k] = filtered_value
        return filtered_dict

    if isinstance(args, (tuple, list)):
       filtered= _filter_iterable(args)
       return _remove_nested_empty_elements(filtered)

    return args

def _remove_nested_empty_elements(data):
    """
    Recursively removes all empty lists or tuples from a nested list or tuple structure.
    :param data: The original list or tuple containing nested lists or tuples.
    :return: The modified list or tuple with empty lists or tuples removed.
    """

    if isinstance(data, (list, tuple)):
        # Recursively process each item and filter out empty lists or tuples
        filtered = [_remove_nested_empty_elements(item) for item in data]

        # Filter out any lists or tuples that are now empty after processing
        filtered = [item for item in filtered if item]

        return type(data)(filtered)
    return data

def _filter_iterable(args):
    '''
    This function recursively iterates over the iterables and filters out any elements that are part of the filter list.
    '''
    if isinstance(args, (tuple, list)):
        return type(args)(_filter_iterable(arg) for arg in args if not isinstance(arg, filter_list))
    return args
