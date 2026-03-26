#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------
# coding: utf-8

# # AIMET Quantization workflow for Phi 4
# 
# This script shows a working code example of how to use AIMET to quantize Phi4 family models

# ---
# ### Required packages
# The script assumes AIMET and Phi4 related packages are already installed.

# Guard to prevent child processes from executing the main script
if __name__ != '__main__':
    import sys
    sys.exit(0)

# ---
# ### Configuration Loading System
# Supports loading configuration from JSON file with 3-tier priority:
# 1. JSON config file (if provided)
# 2. Environment variables
# 3. Default values

import json
import argparse

# Parse command-line arguments for optional config file
parser = argparse.ArgumentParser(
    description='Phi 4 Quantization Script',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Configurable Variables (via JSON config file, environment variables, or defaults):
  MODEL_ID                  Path to the Phi model directory
  CONTEXT_LENGTH            Context length for the model (default: 4096)
  ENABLE_RIGHT_PADDING      Enable right padding of kvcache (default: True)
  APPLY_DECODER_SEQMSE      Apply SeqMSE to decoder (default: True)
  APPLY_LM_HEAD_SEQMSE      Apply SeqMSE to LM head (default: True)
  APPLY_DECODER_LPBQ        Apply LPBQ to decoder (default: False)
  APPLY_LM_HEAD_LPBQ        Apply LPBQ to LM head (default: True)
  ACTIVATION_CLIPPING_CLAMP_VAL  Activation clipping value (default: None)
  EMBEDDING_TABLE_BITWIDTH  Embedding table bitwidth: 8 or 16 (default: 8)
  ENABLE_FP16               Enable FP16 flow (default: False)
  RUN_PPL_EVAL              Run perplexity evaluation (default: True)
  SKIP_PREPARE              Skip model preparation (default: False)
  LD_LIBRARY_PATH           Library path for QNN SDK (default: None)
  TARGET_PLATFORM           Target platform: Windows/Android (default: Windows)
  PLATFORM_GEN              Platform generation: 3/4/5 (default: 3)
  HTP_CONFIG_FILE           Path to HTP quantsim config file (default: auto-detected)
  MODEL_NAME                Model name identifier (default: phi4)
  CACHE_DIR                 Cache directory path (default: ./cache_dir)
  OUTPUT_DIR                Output directory path (default: ./output_dir_phi4)
  NUM_HIDDEN_LAYERS         Number of hidden layers, 0=use model default (default: 0)
  BASE_CALIBRATION_DATASET  Calibration dataset name (default: WIKITEXT)
  ARN                       Auto-regression length (default: 2073)
  MASK_NEG                  Mask negative value (default: -1000)

Priority Order: JSON config > Environment variables > Default values

Example usage:
  python phi4_script.py --config my_config.json
  python phi4_script.py --help
''')
parser.add_argument('--config', type=str, default=None, 
                    help='Path to JSON configuration file')
args, unknown = parser.parse_known_args()

# Load JSON config if provided
json_config = {}
if args.config:
    try:
        with open(args.config, 'r') as f:
            json_config = json.load(f)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Warning: Config file not found: {args.config}")
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in config file: {e}")

def get_config_value(key, default, value_type='str'):
    """
    Get configuration value with 3-tier priority:
    1. JSON config file
    2. Environment variable
    3. Default value
    
    Args:
        key: Configuration key name
        default: Default value if not found in config or environment
        value_type: Type of value ('str', 'int', 'bool', 'none')
    
    Returns:
        Configuration value with appropriate type
    """
    import os
    
    # Priority 1: Check JSON config
    if key in json_config:
        value = json_config[key]
        if value_type == 'bool':
            if isinstance(value, bool):
                return value
            # Handle string boolean values from JSON
            return str(value).lower() in ('true', '1', 't', 'yes')
        elif value_type == 'int':
            return int(value)
        elif value_type == 'none':
            return value  # Can be None or a value
        else:  # str
            return str(value) if value is not None else None
    
    # Priority 2: Check environment variable
    env_value = os.getenv(key)
    if env_value is not None:
        if value_type == 'bool':
            return env_value.lower() in ('true', '1', 't')
        elif value_type == 'int':
            return int(env_value)
        elif value_type == 'none':
            return env_value
        else:  # str
            return env_value
    
    # Priority 3: Use default value
    return default

# ### Overall flow
# This script covers the following
# 1. Parametrizing the Environment
# 2. Instantiate and evaluate FP32 HuggingFace model
# 3. Instantiate and adapt FP32 HuggingFace model
# 4. Model Sample Input
# 5. Prepare model using AIMET model preparer pro
# 6. Evaluation of prepared base model
# 7. Quantization
# 8. Exporting base model onnx, encodings and test vectors
# 
# ### What this script is not 
# * This script is not intended to show the full scope of optimization. For example, the flow will not use QAT, KD-QAT as deliberate choice to have the script execute more quickly.

# ---
# ### 1.1 Notebook Configs

# #### 1.1.1 Notebook Features Config

print("=" * 80)
print("1.1.1 Notebook Features Config")
print("=" * 80)

import os

context_length = get_config_value("CONTEXT_LENGTH", 4096, 'int')

enable_right_padding = get_config_value("ENABLE_RIGHT_PADDING", True, 'bool')  # right padding of kvcache

anchor_alpha = None

# #### 1.1.2 Notebook Quantization Configs

print("=" * 80)
print("1.1.2 Notebook Quantization Configs")
print("=" * 80)

apply_decoder_seqmse = get_config_value("APPLY_DECODER_SEQMSE", True, 'bool')

apply_lm_head_seqmse = get_config_value("APPLY_LM_HEAD_SEQMSE", True, 'bool')

apply_decoder_lpbq = get_config_value("APPLY_DECODER_LPBQ", False, 'bool')

apply_lm_head_lpbq = get_config_value("APPLY_LM_HEAD_LPBQ", True, 'bool')

clamp_val = get_config_value("ACTIVATION_CLIPPING_CLAMP_VAL", None, 'none')

embedding_table_bitwidth = get_config_value("EMBEDDING_TABLE_BITWIDTH", 8, 'int')  # This can be either 8 or 16

# #### 1.1.3 Notebook Configs that will impact Notebook run time

print("=" * 80)
print("1.1.3 Notebook Configs that will impact Notebook run time")
print("=" * 80)

enable_fp16 = get_config_value("ENABLE_FP16", False, 'bool') # Flag to enable e2e fp16 flow, set to false to set fp32 flow

run_ppl_eval = get_config_value("RUN_PPL_EVAL", True, 'bool')

skip_prepare = get_config_value("SKIP_PREPARE", False, 'bool')

assert context_length <= 4096, "Context length longer than 4096 for Phi4 model family has not been validated for accuracy"
assert embedding_table_bitwidth in (8, 16), "Only 8-bit and 16-bit Emebdding Table have been validated"
assert not enable_fp16, "FP16 based quantization has not been tested"

# ---
# ### 1.2 Setting NSP Target

print("=" * 80)
print("1.2 Setting NSP Target")
print("=" * 80)

sys.path.append('../')
from utilities.nsptargets import NspTargets

# setup Target platform and its generation
TARGET_PLATFORM = get_config_value("TARGET_PLATFORM", "Windows").capitalize()

# Android GEN4 and GEN5 is supported for this notebook
PLATFORM_GEN = get_config_value("PLATFORM_GEN", 3, 'int')

nsp_target = eval(f"NspTargets.{TARGET_PLATFORM}.GEN{PLATFORM_GEN}")

# Select quantsim config based on target
htp_config_file = get_config_value(
    'HTP_CONFIG_FILE',
    f'{sys.prefix}/lib/python3.10/site-packages/aimet_common/quantsim_config/htp_quantsim_config_{nsp_target.dsp_arch}.json'
)

# ---
# ## 2. Instantiate and evaluate HuggingFace model

print("=" * 80)
print("2. Instantiate and evaluate HuggingFace model")
print("=" * 80)

import torch
from aimet_torch.utils import place_model, change_tensor_device_placement
from genai_lib.common.debug.profiler import event_marker

os.environ['HF_HOME']="./"
model_name = get_config_value("MODEL_NAME", 'phi4')

model_id = get_config_value("MODEL_ID", "microsoft/Phi-4-reasoning")

cache_dir = get_config_value("CACHE_DIR", './cache_dir')

output_dir = get_config_value("OUTPUT_DIR", f"./output_dir_phi4")

os.makedirs(output_dir, exist_ok=True)

# Note: This cell (and the corresponding cells with Recipe_logger tag) can be removed after dumping and verifying the recipe without 
# impacting notebook functionality
from genai_lib.common.debug.recipe_logger import recipe_dump_init
from genai_lib.common.debug.recipe_logger import llm_lib_log_env_info

# Recipe_logger: Initialize the logger and log environment details 
recipe_dump_init(output_dir)

llm_lib_log_env_info()

# ---
# ### 2.1 Configurable setting by users

print("=" * 80)
print("2.1 Configurable setting by users")
print("=" * 80)

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers.models.phi3 import modeling_phi3

llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
llm_config._attn_implementation="eager"

# To help with debugging num_hidden_layers could be set to 2 to quickly verify the pipeline and export a two layer model for verification purposes
num_hidden_layers = get_config_value("NUM_HIDDEN_LAYERS", 0, 'int')
llm_config.num_hidden_layers = num_hidden_layers if num_hidden_layers > 0 else llm_config.num_hidden_layers

print(f'num_layer: {llm_config.num_hidden_layers}, context_length: {context_length}, '
      f'num_hidden_size: {llm_config.num_attention_heads}, num_kv_heads: {llm_config.num_key_value_heads}')

with event_marker('HuggingFace FP model creation'):
    model = modeling_phi3.Phi3ForCausalLM.from_pretrained(model_id, config=llm_config, cache_dir=cache_dir, trust_remote_code=True)

    os.environ['TOKENIZERS_PARALLELISM'] = '0'
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
    # Adjust the tokenizer to limit to context_length
    tokenizer.model_max_length = context_length

# Reduce the precision of the model to FP16 to minimize the amount of GPU memory needed
if enable_fp16:
    model.half()

# ---
# ### 2.2 Instantiate Dataloaders

print("=" * 80)
print("2.2 Instantiate Dataloaders")
print("=" * 80)

from llm_utils.wikitext_dataloader import get_wiki_dataset

valid_datasets = {}

with event_marker("Instantiate wikitext Dataloaders"):
    wiki_train_dataloader, wiki_test_dataloader, wiki_dataset = get_wiki_dataset(context_length, tokenizer, cache_dir)

valid_datasets["WIKITEXT"] = {
    "dataloader": wiki_train_dataloader,
    "dataset": wiki_dataset
}

base_calibration_key = get_config_value("BASE_CALIBRATION_DATASET", "WIKITEXT").upper()

assert base_calibration_key in valid_datasets, (
   f"`BASE_CALIBRATION_DATASET` must be one of {list(valid_datasets)}, "
   f"but got {base_calibration_key}"
)

base_calibration_dataloader = valid_datasets[base_calibration_key]["dataloader"]
print("Using base calibration dataset:", base_calibration_key)

# ---
# ### 2.3 HuggingFace FP model eval

print("=" * 80)
print("2.3 HuggingFace FP model eval")
print("=" * 80)

from genai_lib.llm.evaluation_utils import llm_evaluate_ppl_with_dataloader
from transformers import pipeline

if run_ppl_eval:
    with event_marker("HuggingFace FP model eval"):
        with place_model(model, torch.device('cuda')):
            orig_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wiki_test_dataloader)

    print(f"PPL score of HuggingFace FP model = {orig_ppl}")

# Remove the HuggingFace model from memory
del model

from genai_lib.common.debug.recipe_logger import llm_lib_log_property, Property
from genai_lib.common.debug.recipe_logger import llm_lib_log_metric, ModelType, Metric

# Recipe_logger: Log the context_length property and the metrics.
llm_lib_log_property({Property.context_length : context_length})

if run_ppl_eval:
    llm_lib_log_metric(ModelType.hf_model, Metric.ppl, orig_ppl, model_name="base")

# ---
# ## 3. Instantiate and adapt FP32 model

# ---
# ### 3.1 Adapt FP32 model definition for inference on HTP.
# - The following adaptations are done to replace default attention module with attention definition that compatible with NSP backend
#   * use conv instead of linear for Q,K,V,O projections
#   * bypass attention and causal mask generation and replace with pre-generated 2D-mask input
#   * output only newly created V and transposed K instead of entire augmented KV sequence
#   * input pre-calculated positional embedding instead of position ids, thus bypass the embedding generation in the model

print("=" * 80)
print("3.1 Adapt FP32 model definition for inference on HTP")
print("=" * 80)

from transformers.models.phi3 import modeling_phi3
from transformers import cache_utils
from genai_lib.common.debug.profiler import event_marker
from genai_lib.llm.dev.model_adaptation.phi.adaptation import (
    QcPhiAttention,
    adapted_update_causal_mask,
    adapted_RotaryEmbedding,
    DynamicCache_update,
    DynamicCache_get_seq_length,
    update_attr,
    QcPhi3ForCausalLM,
    DynamicCache_to_legacy_cache,
    QcPhi3Model
)

with event_marker("FP model adaptation configuration"):
    modeling_phi3.Phi3Attention = QcPhiAttention
    modeling_phi3.Phi3Model = QcPhi3Model
    modeling_phi3.Phi3ForCausalLM = QcPhi3ForCausalLM

    # Bypass attention_mask preparation
    assert hasattr(modeling_phi3.Phi3Model, '_update_causal_mask'), \
    "Phi3Model does not have _update_causal_mask as attribute"
    modeling_phi3.Phi3Model._update_causal_mask = adapted_update_causal_mask

    # Bypass rotary_emb module
    assert hasattr(modeling_phi3.Phi3RotaryEmbedding, 'forward'), \
    f"Unknown Phi3RotaryEmbedding definition: {modeling_phi3.Phi3RotaryEmbedding}"
    modeling_phi3.Phi3RotaryEmbedding.forward = adapted_RotaryEmbedding

    # Adapting KV$ management
    assert update_attr(cache_utils.DynamicCache, 'update', DynamicCache_update), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
    assert update_attr(cache_utils.DynamicCache, 'get_seq_length', DynamicCache_get_seq_length),  f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
    assert update_attr(cache_utils.DynamicCache, 'to_legacy_cache', DynamicCache_to_legacy_cache), \
    f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"

# ---
# ### 3.2 Instantiate adapted FP32 model definition

print("=" * 80)
print("3.2 Instantiate adapted FP32 model definition")
print("=" * 80)

#======================Fixed setting that should not be changed by users==============
# Auto-regression length: number of tokens to consume and number of logits to produce.
# This value should NOT be changed due to downstream consumption requirements
ARN = get_config_value("ARN", 2073, 'int')

enable_right_padding =  enable_right_padding # enable_right_padding is always True when enabled long context length

pad_to_left = not enable_right_padding

setattr(llm_config, 'return_new_key_value_only', True)
setattr(llm_config, 'transposed_key_cache', True)
setattr(llm_config, 'use_combined_mask_input', True)
setattr(llm_config, 'use_position_embedding_input', True)
setattr(llm_config, '_attn_implementation', 'eager')
setattr(llm_config, '_attn_implementation_internal', 'eager')
setattr(llm_config, 'return_dict', False)
setattr(llm_config, 'logits_to_keep', 0)
setattr(llm_config, 'input_tokens_per_inference', ARN)
    
num_slices=(context_length/8 + ARN - 1)//ARN
MASK_NEG = get_config_value("MASK_NEG", -1000, 'int')

llm_config.save_pretrained(output_dir)

from genai_lib.common.debug.recipe_logger import llm_lib_log_property, Property

# Recipe_logger: Log the ARN of the prepared model
llm_lib_log_property({Property.ARN : ARN})

with event_marker('Adapted FP model creation'):
    model = modeling_phi3.Phi3ForCausalLM.from_pretrained(model_id, config=llm_config, cache_dir=cache_dir)

# ---
# ### 3.3 Changes to HuggingFace model to work with the Adapted Model or Prepared Model
# - As a result of adapting the model we introduce changes to the types of the model inputs.
# - As a result of model preparation, we make the shapes of the inputs static.
# - adapted_model_forward works with either adapted model dynamic input or prepared model static input model through flag static_shape.
# - Override the 'forward' function and the function 'prepare_inputs_for_generation'. With these overrides, we make the adapted model or prepared model work just like the old model.
# - adapted_model_prepare_inputs_for_dynamic_shapes is utility function for forward pass of adapted model with dynamic shapes.
# - adapted_model_prepare_inputs_for_static_shapes is utility function for forward pass of prepared model with static shapes.

print("=" * 80)
print("3.3 Changes to HuggingFace model to work with the Adapted Model or Prepared Model")
print("=" * 80)

from genai_lib.llm.static_graph_utils import llm_pad_inputs, llm_create_1d_attn_mask, llm_pad_past_kv, \
    llm_get_position_ids_from_attention_mask, llm_pad_input_attn_mask, llm_create_kv_attn_mask, llm_get_dummy_kv,\
    llm_trim_pad_logits, llm_pad_position_ids,llm_slice_inputs_for_inference
from genai_lib.llm.dev.model_adaptation.phi.utils import llm_update_causal_mask, llm_create_position_embeddings
from genai_lib.llm.dev.model_adaptation.common.utils import KEY_CONCAT_AXIS, VALUE_CONCAT_AXIS, llm_update_kv_cache
from genai_lib.llm.long_context_utils import llm_compute_scores, llm_scatter_exceeded_kv_using_lazy_eviction, llm_update_overwriting_cache
from genai_lib.common.dev.utils import change_signature_defaults
from aimet_torch.utils import change_tensor_device_placement
import types
import random

def get_kv_length(past_key_values):
    if past_key_values is None:
        kv_length = 0
    elif not isinstance(past_key_values, tuple):
        kv_length = past_key_values.get_seq_length()
    else:
        kv_length = past_key_values[0][1].shape[-2]
    return kv_length

def adapted_model_prepare_inputs_for_dynamic_shapes(self,input_ids_slice, attn_mask_slice, position_ids_slice, outputs, **kwargs):
    device = input_ids_slice.device
    batch_size = input_ids_slice.shape[0]
   
    kv_length= get_kv_length(outputs['past_key_values'])

    past_kv_attn_mask = torch.ones((batch_size, kv_length), dtype=torch.long, device=device)
    
    prepared_1d_attention_mask = llm_create_1d_attn_mask(attn_mask_past_kv=past_kv_attn_mask,
                                                         attn_mask_input=attn_mask_slice)

    prepared_causal_mask = llm_update_causal_mask(prepared_1d_attn_mask = prepared_1d_attention_mask,
                                                  input_tensor = input_ids_slice,
                                                  max_input_tokens = input_ids_slice.shape[-1],
                                                  model_context_len = input_ids_slice.shape[-1],
                                                  model_id_or_path = model_id,
                                                  mask_neg = MASK_NEG)

    prepared_position_embeddings = llm_create_position_embeddings(config = llm_config,
                                                                  position_ids = position_ids_slice)
    
    prepared_inputs = {
        'input_ids': input_ids_slice,
        'attention_mask': prepared_causal_mask,
        'position_ids': prepared_position_embeddings,
        'past_key_values': outputs['past_key_values'],
    }

    return prepared_inputs

def adapted_model_prepare_inputs_for_static_shapes(self,input_ids_slice, attn_mask_slice, position_ids_slice, outputs):
    batch_size = input_ids_slice.shape[0]
    pad_token = tokenizer.eos_token_id
    device = input_ids_slice.device
    head_dim = llm_config.head_dim if hasattr(llm_config, 'head_dim') else llm_config.hidden_size // llm_config.num_attention_heads

    if not hasattr(self, "anchor_buffer"):
        shape = (batch_size, llm_config.num_key_value_heads, 1, head_dim)
        self.anchor_buffer = tuple(torch.zeros(shape).to(device = device) for _ in range(llm_config.num_hidden_layers))

    ####### input id preparation #######
    pad_input_ids = llm_pad_inputs(pad_token=pad_token,
                                   max_input_tokens=ARN,
                                   input_ids_slice=input_ids_slice,
                                   pad_to_left=pad_to_left)

    ####### KV input preparation #######
    dummy_kv = llm_get_dummy_kv(batch_size=batch_size,
                                num_key_value_heads=llm_config.num_key_value_heads,
                                head_dim= head_dim,
                                key_concat_axis=KEY_CONCAT_AXIS,
                                device=device,
                                cache_len=context_length-ARN if pad_to_left else context_length)

    padded_past_kv_in = llm_pad_past_kv(dummy_past_kv=dummy_kv,
                                        unpadded_past_kv=outputs['past_key_values'],
                                        num_hidden_layers = llm_config.num_hidden_layers,
                                        key_concat_axis=KEY_CONCAT_AXIS,
                                        value_concat_axis=VALUE_CONCAT_AXIS,
                                        pad_to_left=pad_to_left)

    ######### Attention mask Input preparation #######
    inp_attn_mask = llm_pad_input_attn_mask(attn_mask_slice=attn_mask_slice,
                                            max_input_tokens=ARN,
                                            pad_to_left=pad_to_left)
    
    kv_length = get_kv_length(outputs['past_key_values'])
    
    past_kv_attn_mask = llm_create_kv_attn_mask(unpadded_past_kv= outputs['past_key_values'],
                                                model_context_len=context_length,
                                                max_input_tokens=ARN,
                                                batch_size=batch_size,
                                                device=device,
                                                pad_to_left=pad_to_left)

    if pad_to_left:
        cache_index = None
    else:
        cache_index = torch.tensor([kv_length], dtype=torch.int64, device=device)

    prepared_1d_attention_mask = llm_create_1d_attn_mask(attn_mask_past_kv=past_kv_attn_mask,
                                                         attn_mask_input=inp_attn_mask,
                                                         cache_index=cache_index)

    # due to model adaptation
    prepared_causal_mask = llm_update_causal_mask(prepared_1d_attn_mask = prepared_1d_attention_mask,
                                                  input_tensor = pad_input_ids,
                                                  max_input_tokens = ARN,
                                                  model_context_len = context_length,
                                                  model_id_or_path = model_id,
                                                  cache_index=cache_index,
                                                  pad_to_left= pad_to_left,
                                                  mask_neg = MASK_NEG)

    ########### Position ID preparation #######
    padded_position_ids = llm_pad_position_ids(position_ids_slice=position_ids_slice,
                                                max_input_tokens=ARN, 
                                                pad_to_left = pad_to_left)
    # model adaptation
    prepared_position_embeddings = llm_create_position_embeddings(config = llm_config,
                                                                  position_ids = padded_position_ids)

    prepared_inputs = {
        'input_ids': pad_input_ids,
        'attention_mask': prepared_causal_mask,
        'position_ids': prepared_position_embeddings,
        'past_key_values':padded_past_kv_in
    }

    if enable_right_padding:
        prepared_inputs.update({'cache_index': cache_index})

    return prepared_inputs

from transformers.modeling_outputs import CausalLMOutputWithPast
from genai_lib.llm.static_graph_utils import slice_tensors

# Redefinition of the forward function to work with model I/O adaptations and static shapes of the tensors that the model consumes as input
def adapted_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    position_ids=None,
    return_dict=False,
    inputs_embeds=None,
    output_hidden_states=None,
    **kwargs
):
    static_shape = hasattr(self, 'num_logits_to_return')
    num_slices = kwargs.get('num_slices', None)
    
    if attention_mask is None:
        attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1]), dtype = torch.long, device = input_ids.device)

    if position_ids is None:
        position_ids = torch.cumsum(attention_mask, dim=1) - 1

    # format is: "var_name": (var_tensor, slice_dim)
    inputs = {'input_ids': (input_ids, 1),
              'attention_mask': (attention_mask, 1),
              'position_ids': (position_ids, 1)}
    
    slice_inputs_gen_obj = slice_tensors(slice_length = ARN if static_shape else input_ids.shape[-1],
                                         max_length = input_ids.shape[-1],
                                         tensor_dict = inputs,
                                         remainder_first = True)
    
    # dictionary to store the running output which contains the logits and the useful past kv cache until that execution
    outputs = {}
    outputs['past_key_values'] = past_key_values

    for i, inputs in enumerate(slice_inputs_gen_obj):

        if num_slices is not None and i >= num_slices:
            break
        
        input_ids_slice = inputs['input_ids']
        attn_mask_slice = inputs['attention_mask']
        position_ids_slice = inputs['position_ids']

        if static_shape:  
            prepared_inputs = adapted_model_prepare_inputs_for_static_shapes(self,
                                                                             input_ids_slice=input_ids_slice,
                                                                             attn_mask_slice=attn_mask_slice, 
                                                                             position_ids_slice=position_ids_slice,
                                                                             outputs=outputs)
        else:
            prepared_inputs = adapted_model_prepare_inputs_for_dynamic_shapes(self, 
                                                                              input_ids_slice=input_ids_slice,
                                                                              attn_mask_slice=attn_mask_slice,
                                                                              position_ids_slice=position_ids_slice,
                                                                              outputs=outputs)

        cur_outputs = self.model(**prepared_inputs)
        if not static_shape:
            cur_outputs = (self.lm_head(cur_outputs[0]),) + cur_outputs[1:]
            

        outputs['past_key_values'] = llm_update_kv_cache(unpadded_past_kv = outputs['past_key_values'],
                                                         current_key_values= cur_outputs[1],
                                                         key_concat_axis=KEY_CONCAT_AXIS,
                                                         value_concat_axis=VALUE_CONCAT_AXIS,
                                                         input_ids_slice = input_ids_slice,
                                                         pad_to_left=pad_to_left)

        lm_logits = llm_trim_pad_logits(cur_logits = cur_outputs[0],
                                        input_ids_slice=input_ids_slice,
                                        pad_to_left=pad_to_left)
        bsz, _, dim = lm_logits.shape
        outputs['logits'] = torch.cat(
                (outputs.get('logits', torch.zeros((bsz, 0, dim), device=lm_logits.device)), lm_logits),
                dim=1)

        if output_hidden_states:
            last_hidden_states = llm_trim_pad_logits(cur_logits = cur_outputs[2][-1],
                                                     input_ids_slice=input_ids_slice)
            bsz, _, dim = last_hidden_states.shape
            outputs['hidden_states'] = torch.cat(
                    (outputs.get('hidden_states', torch.zeros((bsz, 0, dim), device=last_hidden_states.device)), last_hidden_states),
                    dim=1)

    if return_dict:
        return CausalLMOutputWithPast(
            loss=outputs.get('loss', None),
            logits=outputs.get('logits', None),
            past_key_values=outputs.get('past_key_values', None),
            hidden_states=outputs.get('hidden_states', None),
            attentions=outputs.get('attentions', None),
        )
    return tuple(outputs.get(out) for out in ['loss', 'logits', 'past_key_values', 'hidden_states', 'attentions'] if outputs.get(out) is not None)

# ---
# ### 3.4 Complete the last step(s) of Model Adaptation
# The following model adaptation are enabled for inference:
# - apply linear to conv in attention, MLP and lmhead and arrange linear weights properly for conv

print("=" * 80)
print("3.4 Complete the last step(s) of Model Adaptation")
print("=" * 80)

from genai_lib.common.dev.model_adaptation.linear_to_conv import ConvInplaceLinear, replace_linears_with_convs
from genai_lib.llm.evaluation_utils import llm_evaluate_ppl_with_dataloader

with event_marker('FP model adaptation for NSP backend completion'):
    #Unpack qkv proj
    for name,module in model.named_modules():
        if isinstance(module, QcPhiAttention):
            module.unpack_qkv()
    
    model = replace_linears_with_convs(model)

if run_ppl_eval:
    model.forward = types.MethodType(adapted_model_forward, model)
    with event_marker(f"Adapted FP model eval"):
        with place_model(model, torch.device('cuda')):
            # model.num_logits_to_return = ARN
            adapted_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wiki_test_dataloader)
    print(f"PPL score of Adapted HF FP model = {adapted_ppl}")

    # Revert forward passes for model preparation
    model.forward = types.MethodType(QcPhi3ForCausalLM.forward, model)

if run_ppl_eval:
    llm_lib_log_metric(ModelType.adapted_model, Metric.ppl, adapted_ppl, model_name="base")

# ---
# ## 4. Model Sample Input

print("=" * 80)
print("4. Model Sample Input")
print("=" * 80)

def get_dummy_data(device="cuda", dtype=torch.float32):
    input_ids = torch.randint(0, len(tokenizer), (1, ARN), device=device)
    attn_mask = torch.ones((1, ARN), device=device, dtype=torch.long)
    position_ids = torch.randint(0, len(tokenizer), (1, ARN), device=device) #1,ARN
    outputs={}
    outputs['past_key_values']=None
    dummy_input = adapted_model_prepare_inputs_for_static_shapes(model, input_ids, attn_mask, position_ids, outputs)
    for val in dummy_input:
        dummy_input[val]= change_tensor_device_placement(dummy_input[val], device)
    # if not return_dict:
    #     dummy_input = tuple(dummy_input.values())
    return dummy_input

# ---
# ## 5. Prepare model using QAIRT model preparer pro

# ---
# ### 5.1 KVCache MHA model preparation

# ##### Fix LazyQuantizeWrapper attribute delegation
# Monkey patch in AIMET to fix exception rule failure with nonleaf qmodules, it ensures attributes not found in the wrapper are properly delegated to the wrapped module

print("=" * 80)
print("5.1 KVCache MHA model preparation")
print("=" * 80)

import time
from qti.aisw.emitter.utils.torch_utils import load_torch_model_using_safetensors
from genai_lib.llm.model_preparation_utils import llm_build_preparer_converter_args
from genai_lib.llm.utils import llm_model_input_output_names
from qti.aisw.preparer_api.model_preparer import prepare_model

# Configuring the model for KVCache mode
model.num_logits_to_return = ARN

prepare_path = os.path.join(output_dir, 'prepare')
os.makedirs(prepare_path, exist_ok=True)
prepare_filename = f'{model_name}_kvcache_{llm_config.num_hidden_layers}_layer'

if not skip_prepare:
    dummy_input = get_dummy_data(device=model.model.device)
    input_names, output_names = llm_model_input_output_names(llm_config.num_hidden_layers)
    if enable_right_padding:
        input_names += ["cache_index"]

    converter_args = llm_build_preparer_converter_args(llm_config.num_hidden_layers, input_names, use_qairt_mpp=True) # Build converter args

    with event_marker("Prepare Model", flush_ram=True):

        if __name__ == '__main__': # We use the main guard to prevent child processes from re-running the top-level code
            _ = prepare_model(model,
                              dummy_input,
                              model_name = prepare_filename,
                              filename = prepare_filename,
                              path = prepare_path,
                              input_names = input_names,
                              output_names = output_names,
                              onnx_export_args = {"opset_version":17},
                              converter_args = converter_args,
                              keep_original_model_structure = False, # Flatten the model to enable weight-sharing by setting
                              order_inputs = True,
                              order_outputs = True,
                              skipped_optimizers = ['eliminate_common_subexpression',
                                                   'eliminate_nop_with_unit',
                                                   'eliminate_duplicate_initializer'
                                                   ],
                               return_prepare_model = False
                               )

# ---
# ## 6. Evaluation of prepared model
# Verify if prepared KV cache model generates the same PPL as FP model.

# ---
# ### 6.1 Changes to HuggingFace model to work with the prepared model
# 
# Replace the model inside the HuggingFace model with the prepared model.
# Note that the prepared model already fuses model.model and model.lm_head 
# into one, so here we simply set model.lm_head to None

print("=" * 80)
print("6.1 Changes to HuggingFace model to work with the prepared model")
print("=" * 80)

del model.model
del model.lm_head

model.model = None
model.lm_head = None

with event_marker(f"Load pre-prepared {prepare_filename}", flush_ram=True):
    prepared_model_path = os.path.join(prepare_path, f'{prepare_filename}.py')
    if not os.path.exists(prepared_model_path):
        raise ValueError(f"prepared artifacts not found in {prepare_path}")
    elif skip_prepare:
        print(f'Preparation skipped for model={prepare_filename}, prepared at {time.ctime(os.path.getmtime(prepared_model_path))}')
    prepared_model = load_torch_model_using_safetensors(path=prepare_path, filename=prepare_filename, model_name=prepare_filename)

model.model = prepared_model
model.forward = types.MethodType(adapted_model_forward, model)

# ---
# ### 6.2 Convert the model to half precision

print("=" * 80)
print("6.2 Convert the model to half precision")
print("=" * 80)

if enable_fp16:
    torch.set_default_dtype(torch.float16)
    model.half()

# ---
# ### 6.3 Evaluation of perplexity score using prepared model

print("=" * 80)
print("6.3 Evaluation of perplexity score using prepared model")
print("=" * 80)

if run_ppl_eval:
    with event_marker("KVcache prepared FP eval", flush_ram=True):
        with place_model(model, torch.device("cuda")):
            prepared_kvcache_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wiki_test_dataloader)

    # This should be very close (<1e-4 delta) to original model's perplexity
    # If the perplexity score goes further up, it indicates the AIMET/QNN pair is producing a faulty prepared model
    print(f"ppl score of KVCACHE prepared fp model = {prepared_kvcache_ppl}")
    print(f"Diff between HF orig ppl and prepared ppl = {orig_ppl - prepared_kvcache_ppl}")

if run_ppl_eval:
    llm_lib_log_metric(ModelType.prepared_model, Metric.ppl, prepared_kvcache_ppl, model_name="base")

# ---
# ## 7. Quantization
# 
# The _Quantization_ step is the primary focus of this notebook, this section could be modified to execute various quantization experiments.

# ---
# ### 7.1 Create quantsim configured for QNN HTP target 

print("=" * 80)
print("7.1 Create quantsim configured for QNN HTP target")
print("=" * 80)

from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel
import aimet_common.quantsim as qs
import inspect
from copy import deepcopy

qs.encoding_version = '1.0.0'

if apply_lm_head_seqmse or apply_decoder_seqmse:
    import functools

    def copy_model_with_shared_weights(source_model):
        target_model = deepcopy(source_model)
        for name, source_parameter in source_model.named_parameters():
            pre, _, post = name.rpartition('.')
            pre_obj = functools.reduce(getattr, [target_model] + pre.split('.')) if pre else target_model
            setattr(pre_obj, post, source_parameter)
        return target_model

    # Create copy of fp model defintion for SeqMSE and/or LoRA
    fp_prepared_model = copy_model_with_shared_weights(prepared_model)

dummy_input = get_dummy_data(device = "cuda", dtype = model.dtype)

sig = inspect.signature(prepared_model.forward)
dummy_input_sorted = {}
for key in list(sig.parameters.keys()):
    dummy_input_sorted[key] = dummy_input[key]
dummy_input = tuple(dummy_input_sorted.values())

with event_marker("create KVCache Quantsim"):
    with place_model(prepared_model, "cuda"):
        quantsim = QuantizationSimModel(model=prepared_model,
                                        quant_scheme=QuantScheme.post_training_tf,
                                        dummy_input=dummy_input,
                                        default_output_bw=16,
                                        default_param_bw=4,
                                        in_place=True,
                                        config_file=htp_config_file)

# ---
# ### 7.2 Setting 16bit x 8bit matmuls
# To keep key and value tensors as 8 bits, reducing data I/O costs associated with KV-cache orchestration.

print("=" * 80)
print("7.2 Setting 16bit x 8bit matmuls")
print("=" * 80)

from aimet_torch.v2.experimental.quantsim_utils import set_matmul_second_input_producer_to_8bit_symmetric

set_matmul_second_input_producer_to_8bit_symmetric(quantsim)

# ---
# ### 7.3 Concat encoding unification
# configuring concat ops to have shared encoding on input and output activations.

print("=" * 80)
print("7.3 Concat encoding unification")
print("=" * 80)

from aimet_torch.v2.experimental import propagate_output_encodings
from aimet_torch.nn.modules import custom as aimet_ops

propagate_output_encodings(quantsim, aimet_ops.Concat)

# ---
# ### 7.4 Manual Mixed Precision
# applying mixed precision configuration to ops 

print("=" * 80)
print("7.4 Manual Mixed Precision")
print("=" * 80)

import json

with open("./config/mixed_precision_config/exceptions.json", "r") as f_in:
    mixed_precision_config = json.load(f_in)

# Customize mixed precision config based on user parameters
for entry in mixed_precision_config['name_list']:
    if "model_embed_tokens_Gather" in entry['module_name']:
        entry['exceptions']['param_exceptions']['bitwidth'] = embedding_table_bitwidth

from llm_utils.mixed_precision_overrides import ManualQuantsimMixedPrecisionConfig

quantsim_adjuster = ManualQuantsimMixedPrecisionConfig(mixed_precision_config_file = mixed_precision_config)
quantsim_adjuster.apply_exceptions(quantsim)

from aimet_torch.v2.nn.modules.custom import QuantizedRmsNorm
from aimet_torch.v2.quantization.affine import QuantizeDequantize

# Make RMSNorm encodings per-tensor (they default to per-channel)
for name, qmodule in quantsim.named_qmodules():
    if isinstance(qmodule, QuantizedRmsNorm):
        qmodule.param_quantizers['weight'] = QuantizeDequantize(shape=(), bitwidth=16, symmetric=False).to(qmodule.weight.device)

# ---
# ### 7.5 Apply Block Quantization
# Swapping needed modules' weight quantizers to LPBQ quantizers

print("=" * 80)
print("7.5 Apply Block Quantization")
print("=" * 80)

from aimet_torch.v2.nn.true_quant import QuantizedConv2d
from aimet_torch.v2.quantsim.config_utils import set_grouped_blockwise_quantization_for_weights

arg = None

if apply_decoder_lpbq and apply_lm_head_lpbq:
    arg = lambda module: isinstance(module, QuantizedConv2d)
elif apply_decoder_lpbq:
    arg = lambda module: isinstance(module, QuantizedConv2d) and module.param_quantizers['weight'].bitwidth == 4
elif apply_lm_head_lpbq:
    lm_head_modules = [qmodule for name, qmodule in quantsim.named_qmodules() if "lm_head" in name]
    arg = lambda module: module in lm_head_modules and isinstance(module, QuantizedConv2d)
    
if arg:
    BLOCK_QUANT_SIZE = 128
    set_grouped_blockwise_quantization_for_weights(sim = quantsim,
                                                   arg = arg,
                                                   bitwidth = 4,
                                                   symmetric = True,
                                                   decompressed_bw = 8,
                                                   block_size = BLOCK_QUANT_SIZE,
                                                   block_grouping = -1)

### Unify past_key/value_{x}_out encodings (input[2], input[0], output[0]) to upstream Ops  (self_attn_Concat_1/self_attn_v_proj_Conv)

def unify_scatter_elements_encodings(source_name, destination_name):

    def _find_module_dict(name):
        for module_name, module in quantsim.model.named_modules():
            if module_name.endswith(name):
                start = module_name.find(name)
                yield module_name[:start], module

    sources = { name:module for name, module in _find_module_dict(source_name) }
    destinations = { name:module for name, module in _find_module_dict(destination_name) }

    assert len(sources)==len(destinations) and len(sources)> 0, f"Cannot execute encoding alignment due to mismatched pairing of \
    source and destination quantizers. String matching found {len(sources)} sources, and {len(destinations)} destinations."
    # copying quantizers from source module 
    for module_name, source_module in sources.items():
        desination_module = destinations[module_name]
        desination_module.input_quantizers[2]=source_module.output_quantizers[0]
        desination_module.input_quantizers[0]=source_module.output_quantizers[0]
        desination_module.output_quantizers[0]=source_module.output_quantizers[0]        

if enable_right_padding:
    unify_scatter_elements_encodings('self_attn_Concat_1', 'self_attn_ScatterElements_1')
    unify_scatter_elements_encodings('self_attn_v_proj_Conv', 'self_attn_ScatterElements')

# ---
# ### 7.7 Sequential MSE
# applying sequential MSE technique to optimize parameter encodings

print("=" * 80)
print("7.7 Sequential MSE")
print("=" * 80)

def _seq_mse_forward_fn(_model, inputs):
    model.model = _model
    model(**inputs, num_slices=num_slices)

if apply_decoder_seqmse or apply_lm_head_seqmse:
    from aimet_torch.v2.seq_mse import apply_seq_mse, SeqMseParams

    lm_head_fp_modules = [ module for module_name, module in fp_prepared_model.named_modules() if isinstance(module, torch.nn.Conv2d) and 'lm_head' in module_name ]    
    decoder_fp_modules = [ module for module_name, module in fp_prepared_model.named_modules() if isinstance(module, torch.nn.Conv2d) and 'lm_head' not in module_name ]

    if apply_decoder_seqmse and apply_lm_head_seqmse:
        modules_to_exclude = []
    elif apply_decoder_seqmse:
        modules_to_exclude = lm_head_fp_modules
    elif apply_lm_head_seqmse:
        modules_to_exclude = decoder_fp_modules

    params = SeqMseParams(num_batches=20,
                          inp_symmetry='symqt',
                          num_candidates=20,
                          loss_fn='mse',
                          forward_fn = _seq_mse_forward_fn)

    with event_marker("SeqMSE"):
        with place_model(quantsim.model, torch.device("cuda")), place_model(fp_prepared_model, torch.device("cuda")):
            with torch.no_grad():
                apply_seq_mse(fp_prepared_model, quantsim, wiki_train_dataloader, params, modules_to_exclude=modules_to_exclude)

    del fp_prepared_model

# ---
# ### 7.8 Calibration

print("=" * 80)
print("7.8 Calibration")
print("=" * 80)

from tqdm import tqdm
from aimet_torch.v2.experimental.quantsim_utils import clip_weights_to_7f7f

def _calibration_forward_fn(sim_model, kwargs):

    model.model = sim_model
    data_loader = kwargs['data_loader']
    max_iterations = kwargs['num_batches']
    for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
        if batch_id < max_iterations:
            model(input_ids=batch['input_ids'].to(device=torch.device('cuda')), 
                    num_slices=num_slices)
        else:
            break

kwargs = {
    'data_loader': base_calibration_dataloader,
    'num_batches': 200
}

with event_marker("compute encoding", flush_ram=True):
    with place_model(quantsim.model, "cuda"):
        with torch.no_grad():
            quantsim.compute_encodings(_calibration_forward_fn, kwargs)

clip_weights_to_7f7f(quantsim)

# ---
# ### 7.9 Apply Activation Clipping

print("=" * 80)
print("7.9 Apply Activation Clipping")
print("=" * 80)

def apply_clipping(quantsim, clamp_val):
    from aimet_torch.v2.nn.base import BaseQuantizationMixin as QUANTIZED_MODULE

    def _clip_and_recompute_encodings(quantizer, name, clamp_val):
        if not quantizer.is_initialized():
            return
        qmin = quantizer.min.min()
        qmax = quantizer.max.max()
        if qmin < -clamp_val or qmax > clamp_val:
            quantizer.min.data = torch.clamp(quantizer.min, -clamp_val, clamp_val)
            quantizer.max.data = torch.clamp(quantizer.max, -clamp_val, clamp_val)

            print(f"{name} activation clamping... before: {qmin}, {qmax} | after: {quantizer.min.min().item()}, {quantizer.max.max().item()}")

    # Apply activation clipping
    for name, module in quantsim.model.named_modules():
        if isinstance(module, QUANTIZED_MODULE):
            for quantizer in module.output_quantizers:
                if quantizer:
                    _clip_and_recompute_encodings(quantizer, name + " | output quantizer", clamp_val)
            for quantizer in module.input_quantizers:
                if quantizer:
                    _clip_and_recompute_encodings(quantizer, name + " | input quantizer", clamp_val)

if clamp_val is not None:
    apply_clipping(quantsim, int(clamp_val))

# ---
# ### 7.10 Eval KV Cache sim on Base Model

print("=" * 80)
print("7.10 Eval KV Cache sim on Base Model")
print("=" * 80)

if run_ppl_eval:
    with event_marker("KV cache sim with base model eval", flush_ram=True):
        with place_model(quantsim.model, torch.device("cuda")):
            model.model = quantsim.model
            sim_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wiki_test_dataloader)

    print(f"ppl score of KVCACHE sim with base model = {sim_ppl}")
    print(f"Diff between orig ppl and kvcache sim ppl = {orig_ppl - sim_ppl}")

if run_ppl_eval:
    # Recipe_logger: Log the ppl for qsim model and dump the cumulative logs to a JSON file.
    llm_lib_log_metric(ModelType.qsim_model, Metric.ppl, sim_ppl, model_name="base")

# ---
# ## 8. Export
# the pipeline call below would export onnx model, encodings and test vector for KVCache model.

# ---
# ### 8.1 Export Onnx and Encodings

print("=" * 80)
print("8.1 Export Onnx and Encodings")
print("=" * 80)

from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch import onnx_utils

# Get input names and output names. This is different from the input names and output names we created for model preparation. 
# The reason for this difference stems from the fact that we want the prepared model to have inputs and outputs named similar to original HF model
# ONNX does not allow tupling the inputs or outputs and we want to give meaningful names to the input and output tensors in the ONNX graph
input_names, output_names = llm_model_input_output_names(llm_config.num_hidden_layers, use_position_embedding_input=True, separate_tuple_input_output=True)

def _get_anchor_buffer_names(sfx, n_layers):
    return [f'anchor_buffer_{i}_{sfx}' for i in range(n_layers)]
    

if enable_right_padding:
    input_names += ["cache_index"]

if enable_fp16:
    # Convert FP16 model back to FP32 for ONNX export
    torch.set_default_dtype(torch.float32)
    model.float()

onnx_api_args = OnnxExportApiArgs(input_names=input_names, output_names=output_names, opset_version=17)

base_filename_prefix = f"{model_name}_base"

onnx_utils.EXPORT_TO_ONNX_DIRECT = True

onnx_utils.RESTORE_ONNX_MODEL_INITIALIZERS = True

dummy_input = get_dummy_data(device = "cpu", dtype = model.dtype)

base_onnx_dir = os.path.join(output_dir, 'base', 'onnx')
os.makedirs(base_onnx_dir, exist_ok=True)

sig = inspect.signature(prepared_model.forward)
dummy_input_sorted = {}
for key in list(sig.parameters.keys()):
    dummy_input_sorted[key] = dummy_input[key]
dummy_input = dummy_input_sorted
dummy_input = tuple(list(dummy_input.values()))

with event_marker(f"KVCache export onnx and encodings", flush_ram=True):
    with torch.no_grad():
        with place_model(quantsim.model, torch.device("cpu")):
            quantsim.export(base_onnx_dir, base_filename_prefix, dummy_input, onnx_export_args=onnx_api_args,filename_prefix_encodings=base_filename_prefix)

# Exporting Tokenizer
tokenizer.save_pretrained(output_dir)

# Export chat template
if getattr(tokenizer, "chat_template", None):
    with open(os.path.join(output_dir, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(tokenizer.chat_template)
else:
    print("No chat_template found on tokenizer; nothing to export.")

# Export generation config
model.generation_config.save_pretrained(output_dir)

# ---
# ### 8.2 Generating test vectors for QNN SDK

print("=" * 80)
print("8.2 Generating test vectors for QNN SDK")
print("=" * 80)

from genai_lib.llm.test_vectors import generate_test_vectors

test_vector_layers = [
    "model_embed_tokens_Gather",
    "model_layers_\\d+_Add_1"
]

num_test_vectors = 1

with event_marker("generate base model test vectors"):
    with place_model(quantsim.model, torch.device("cuda")):
        for index, batch in enumerate(wiki_train_dataloader):
            if index >= num_test_vectors:
                break
            input_ids_slice = batch['input_ids'][..., :ARN].to(device=torch.device('cuda'))
            attn_mask_slice = torch.ones((input_ids_slice.shape[0], ARN), dtype=torch.long, device=torch.device('cuda'))
            position_ids_slice = torch.cumsum(attn_mask_slice, dim=1) - 1
            outputs = {'past_key_values': None}
            model_inputs = adapted_model_prepare_inputs_for_static_shapes(model, input_ids_slice=input_ids_slice, 
                                                                          attn_mask_slice=attn_mask_slice, 
                                                                          position_ids_slice=position_ids_slice,
                                                                          outputs=outputs)
            generate_test_vectors(sim=quantsim, model_inputs=model_inputs, output_dir=os.path.join(output_dir, 'base'), batch_index=index, test_vector_layers=test_vector_layers)

# ---
# ### Summary

print("=" * 80)
print("Summary")
print("=" * 80)

from genai_lib.common.debug.profiler import EventProfiler
from genai_lib.common.debug.recipe_logger import dump_logs_to_json

EventProfiler().report()
EventProfiler().json_dump(os.path.join(output_dir, 'profiling_stats.json'))
dump_logs_to_json()

print("=" * 80)
print("Script End")
print("=" * 80)