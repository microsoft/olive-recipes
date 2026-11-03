#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------
# coding: utf-8

# # AIMET Quantization workflow for Llama 3.1 8B with AdaScale
#
# This script shows a working code example of how to use AIMET to:
# 1. Apply AdaScale optimization to Llama 3.1 model weights
# 2. Quantize the AdaScale-optimized model for deployment

try:
    # Required for proper Python environment configuration of qairt-dev
    import qairt  # noqa: F401  # pylint: disable=unused-import
except ImportError as exc:
    raise ImportError(
        "Failed to import QAIRT SDK - please install olive-ai[qairt] to use QAIRT passes."
        "If already installed, please run `qairt-vm -i` for help troubleshooting issues."
    ) from exc

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
import os
from tqdm import tqdm

# Parse command-line arguments for optional config file
parser = argparse.ArgumentParser(
    description='Phi4 mini Instruct AdaScale + Quantization Script',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Configurable Variables (via JSON config file, environment variables, or defaults):

AdaScale Phase:
  MODEL_ID                      Path to the initial Phi4 mini instruct model
  ADASCALE_CONTEXT_LENGTH       Context length for AdaScale (default: 2048)
  ADASCALE_ITERATIONS           Number of AdaScale iterations (default: 5000)
  ENABLE_BF16                   Enable BF16 for AdaScale (default: False)
  NUM_EVAL_BATCHES              Number of batches for evaluation (default: 0)
  C4_DATASET_PATH               Path to C4 dataset JSON (auto-downloads if not provided)
  BATCH_SIZE                    Batch size for AdaScale dataloader (default: 2)
  PERCENT_DATASET_TO_LOAD       Percentage of C4 dataset to load (default: 1)
  NUM_SAMPLES                   Number of samples from C4 (default: 500)
  ADASCALE_KEEP_OUTPUTS         Keep AdaScale intermediate outputs (default: False)

Base Quantization Phase:
  ENABLE_RIGHT_PADDING          Enable right padding of kvcache (default: True)
  APPLY_DECODER_SEQMSE          Apply SeqMSE to decoder (default: False)
  APPLY_LM_HEAD_SEQMSE          Apply SeqMSE to LM head (default: False)
  APPLY_DECODER_LPBQ            Apply LPBQ to decoder (default: False)
  APPLY_LM_HEAD_LPBQ            Apply LPBQ to LM head (default: True)
  ACTIVATION_CLIPPING_CLAMP_VAL Activation clipping value (default: None)
  EMBEDDING_TABLE_BITWIDTH      Embedding table bitwidth: 8 or 16 (default: 8)
  ENABLE_FP16                   Enable FP16 flow (default: False)
  WIKI_DATASET_PATH             Path to wikitext dataset (optional)

Shared:
  TARGET_PLATFORM               Target platform: Windows/Android (default: Windows)
  PLATFORM_GEN                  Platform generation: 2/3/4/5 (default: 3)
  RUN_PPL_EVAL                  Run perplexity evaluation (default: True)
  MODEL_NAME                    Model name identifier (default: llama3_1_instruct)
  CACHE_DIR                     Cache directory path (default: ./cache_dir)
  OUTPUT_DIR                    Output directory path (default: ./output_dir)
  NUM_HIDDEN_LAYERS             Number of hidden layers, 0=use model default (default: 0)
  BASE_CALIBRATION_DATASET      Calibration dataset name (default: WIKITEXT)

Priority Order: JSON config > Environment variables > Default values

Example usage:
  python phi4_mini_instruct_script.py --config my_config.json
  python phi4_mini_instruct_script.py --help
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
    # Priority 1: Check JSON config
    if key in json_config:
        value = json_config[key]
        if value_type == 'bool':
            if isinstance(value, bool):
                return value
            return str(value).lower() in ('true', '1', 't', 'yes')
        elif value_type == 'int':
            return int(value)
        elif value_type == 'none':
            return value
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

# Helper function to download C4 dataset if needed for AdaScale
def download_c4_dataset_if_needed(cache_dir):
    """
    Download C4 dataset if not already present in cache_dir/c4-dataset/

    Args:
        cache_dir: Base cache directory path

    Returns:
        Path to the C4 dataset JSON file
    """
    import urllib.request
    import gzip
    import shutil

    c4_dir = os.path.join(cache_dir, "c4-dataset")
    c4_filename = "c4-train.00000-of-01024.json"
    c4_file = os.path.join(c4_dir, c4_filename)
    c4_gz_file = c4_file + ".gz"

    # Check if file already exists
    if os.path.exists(c4_file):
        print(f"C4 dataset found at: {c4_file}")
        return c4_file

    print("=" * 80)
    print("Downloading C4 dataset for AdaScale")
    print("=" * 80)

    # Create directory if it doesn't exist
    os.makedirs(c4_dir, exist_ok=True)

    # Download URL
    c4_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"

    try:
        # Download the compressed file
        print(f"Downloading from: {c4_url}")
        print(f"Saving to: {c4_gz_file}")
        print("This may take several minutes depending on your connection...")

        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        urllib.request.urlretrieve(c4_url, c4_gz_file, reporthook=download_progress)
        print("\nDownload complete!")

        # Decompress the file
        print(f"Decompressing {c4_filename}.gz...")
        with gzip.open(c4_gz_file, 'rb') as f_in:
            with open(c4_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the compressed file
        os.remove(c4_gz_file)
        print(f"Decompression complete! File saved at: {c4_file}")

        return c4_file

    except Exception as e:
        print(f"\nError downloading C4 dataset: {e}")
        print("Please download manually using:")
        print(f"  wget {c4_url}")
        print(f"  gunzip {c4_filename}.gz")
        print(f"  mv {c4_filename} {c4_dir}/")
        raise

# ============================================================================
# PART 1: ADASCALE OPTIMIZATION
# ============================================================================

print("=" * 80)
print("PART 1: ADASCALE OPTIMIZATION")
print("=" * 80)

# ---
# ### 1.1 AdaScale Notebook Configs

print("=" * 80)
print("1.1 AdaScale Notebook Configs")
print("=" * 80)

import sys

# Feature knobs
adascale_context_length = get_config_value("ADASCALE_CONTEXT_LENGTH", 2048, 'int')

# Quantization knobs
adascale_iterations = get_config_value("ADASCALE_ITERATIONS", 5000, 'int')

# Speed knobs
run_ppl_eval = get_config_value("RUN_PPL_EVAL", True, 'bool')
enable_bf16 = get_config_value("ENABLE_BF16", False, 'bool')
num_eval_batches = get_config_value("NUM_EVAL_BATCHES", 0, 'int')

# ---
# ### 1.2 Setting NSP Target for AdaScale

print("=" * 80)
print("1.2 Setting NSP Target for AdaScale")
print("=" * 80)

from utilities.nsptargets import NspTargets

os.environ['HF_HOME']="./"

# setup Target platform and its generation
TARGET_PLATFORM = get_config_value("TARGET_PLATFORM", "Windows").capitalize()

# Android GEN4 and GEN5 is supported for this notebook
PLATFORM_GEN = get_config_value("PLATFORM_GEN", 2, 'int')

nsp_target_adascale = eval(f"NspTargets.{TARGET_PLATFORM}.GEN{PLATFORM_GEN}")

# Select quantsim config based on target
htp_config_file_adascale = f'htp_quantsim_config_v81.json'

# ---
# ## 2. Instantiate and evaluate HuggingFace model for AdaScale

print("=" * 80)
print("2. Instantiate and evaluate HuggingFace model for AdaScale")
print("=" * 80)

import torch
#from transformers import AutoModelForCausalLM
from transformers.models.phi3 import modeling_phi3
from aimet_torch.utils import place_model, change_tensor_device_placement
from genai_lib.common.debug.profiler import event_marker

model_name = get_config_value("MODEL_NAME", 'phi4_mini_instruct')
model_id = get_config_value("MODEL_ID", "microsoft/phi-4-mini-instruct")

cache_dir = get_config_value("CACHE_DIR", './cache_dir')
output_dir = get_config_value("OUTPUT_DIR", "./output_dir")
adascale_dir = os.path.join(output_dir, "adascale_output")

os.makedirs(adascale_dir, exist_ok=True)


# Recipe_logger: Initialize the logger and log environment details
from genai_lib.common.debug.recipe_logger import llm_lib_log_env_info, recipe_dump_init

recipe_dump_init(adascale_dir, "genai_lib_debug")
llm_lib_log_env_info()

# ---
# #### 2.1 Load HF Model for AdaScale

print("=" * 80)
print("2.1 Load HF Model for AdaScale")
print("=" * 80)

from transformers import AutoConfig, AutoTokenizer

llm_config_adascale = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
num_hidden_layers = get_config_value("NUM_HIDDEN_LAYERS", 0, 'int')
llm_config_adascale.num_hidden_layers = num_hidden_layers if num_hidden_layers > 0 else llm_config_adascale.num_hidden_layers

print(f'num_layer: {llm_config_adascale.num_hidden_layers}, context_length: {adascale_context_length}, '
      f'num_hidden_size: {llm_config_adascale.num_attention_heads}, num_kv_heads: {llm_config_adascale.num_key_value_heads}')

with event_marker('HuggingFace FP model creation for AdaScale'):
    model_adascale = modeling_phi3.Phi3ForCausalLM.from_pretrained(model_id, config=llm_config_adascale, cache_dir=cache_dir,
                                                           torch_dtype=torch.bfloat16 if enable_bf16 else torch.float32)

    os.environ['TOKENIZERS_PARALLELISM'] = '0'
    tokenizer_adascale = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
    # Adjust the tokenizer to limit to context_length
    tokenizer_adascale.model_max_length = adascale_context_length

# ---
# #### 2.2 Instantiate Dataloaders for AdaScale

print("=" * 80)
print("2.2 Instantiate Dataloaders for AdaScale")
print("=" * 80)

from llm_utils.wikitext_dataloader import get_wiki_dataset
from llm_utils.generic_dataloader import get_local_dataset

with event_marker("Instantiate wikitext dataloaders for AdaScale"):
    _, wikitext_test_dataloader_adascale, _ = get_wiki_dataset(adascale_context_length, tokenizer_adascale, cache_dir)

# Download C4 dataset if needed
c4_dataset_path = get_config_value("C4_DATASET_PATH", None)
if c4_dataset_path is None:
    c4_dataset_path = download_c4_dataset_if_needed(cache_dir)

with event_marker("Instantiate adascale dataloaders"):
    adascale_train_dataloader, _ = get_local_dataset(
        adascale_context_length,
        tokenizer_adascale,
        json_path=c4_dataset_path,
        key="input_ids",
        batch_size=get_config_value("BATCH_SIZE", 2, 'int'),
        percent_dataset_to_load=get_config_value("PERCENT_DATASET_TO_LOAD", 1, 'int'),
        num_samples=get_config_value("NUM_SAMPLES", 500, 'int')
    )

# ---
# #### 2.3 Eval HF Model for AdaScale

print("=" * 80)
print("2.3 Eval HF Model for AdaScale")
print("=" * 80)

from genai_lib.llm.evaluation_utils import llm_evaluate_ppl_with_dataloader

if run_ppl_eval:
    with event_marker("HuggingFace FP model eval for AdaScale"):
        with place_model(model_adascale, torch.device('cuda')):
            orig_ppl_adascale = llm_evaluate_ppl_with_dataloader(model=model_adascale, dataloader=wikitext_test_dataloader_adascale,
                                                                  num_batches=num_eval_batches)
    print(f"PPL score of HuggingFace FP model (AdaScale) = {orig_ppl_adascale}")

from genai_lib.common.debug.recipe_logger import llm_lib_log_property, Property
from genai_lib.common.debug.recipe_logger import llm_lib_log_metric, ModelType, Metric

# Recipe_logger: Log the context_length property and the metrics.
llm_lib_log_property({Property.context_length: adascale_context_length})

if run_ppl_eval:
    llm_lib_log_metric(ModelType.hf_model, Metric.ppl, orig_ppl_adascale)

# ---
# ## 3. AdaScale

print("=" * 80)
print("3. AdaScale")
print("=" * 80)

# ---
# #### 3.1 Redefine forward for JIT tracing in Quantsim Creation

print("=" * 80)
print("3.1 Redefine forward for JIT tracing in Quantsim Creation")
print("=" * 80)

from transformers import PreTrainedModel, DynamicCache
import types

# AIMET requires KV Cache to be of type Tuple during the forward pass, so we wrap the forward to convert the KV Cache during inference
def custom_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, *args, **kwargs):
    past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    lm_logits, new_past_key_values = self.__original_forward__(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        num_logits_to_return=0,
        return_dict=False,
        *args,
        **kwargs,
    )

    return lm_logits, new_past_key_values.to_legacy_cache()

# Save original forward method
model_adascale.__original_forward__ = model_adascale.forward
# Replace with custom forward
model_adascale.forward = types.MethodType(custom_forward, model_adascale)

# ---
# #### 3.2 Create quantsim configured for QNN HTP target

print("=" * 80)
print("3.2 Create quantsim configured for QNN HTP target")
print("=" * 80)

from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel
#from aimet_torch import QuantizationSimModel

dummy_input_adascale = torch.randint(0, 1, (1, 1, adascale_context_length), device="cuda")

with event_marker("create KVCache Quantsim for AdaScale"):
    with place_model(model_adascale, "cuda"):
        model_adascale.config.return_dict = False
        quantsim_adascale = QuantizationSimModel(model=model_adascale,
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  default_output_bw=16,
                                                  default_param_bw=4,
                                                  in_place=True,
                                                  dummy_input=tuple(list(dummy_input_adascale)),
                                                  config_file=htp_config_file_adascale)

from aimet_torch.v2.experimental import propagate_output_encodings
from aimet_torch.nn.modules import custom as aimet_ops

propagate_output_encodings(quantsim_adascale, aimet_ops.Concat)

# ---
# #### 3.3 Enable per channel quantization

print("=" * 80)
print("3.3 Enable per channel quantization")
print("=" * 80)

from aimet_torch.v2.nn.true_quant import QuantizedLinear
from aimet_torch.v2.quantization.affine import QuantizeDequantize

for name, qmodule in quantsim_adascale.named_qmodules():
    if isinstance(qmodule, QuantizedLinear):
        assert (len(qmodule.weight.shape)) == 2, f"Per channel quantization for linear weights is only supported for 2d weights, got shape: {qmodule.weight.shape} instead"
        qmodule.param_quantizers["weight"] = QuantizeDequantize(shape=(qmodule.weight.shape[0], 1),
                                                                bitwidth=qmodule.param_quantizers["weight"].bitwidth,
                                                                symmetric=qmodule.param_quantizers["weight"].symmetric).to(next(quantsim_adascale.model.parameters()).device)

# ---
# #### 3.4 Manual mixed precision + Disable un-needed quantizers

print("=" * 80)
print("3.4 Manual mixed precision + Disable un-needed quantizers")
print("=" * 80)

import re

# Remove quantizers for non decoder blocks
quantsim_adascale.model.model.embed_tokens.param_quantizers["weight"] = None
quantsim_adascale.model.lm_head.param_quantizers["weight"] = None

# Increase bitwidth for rmsnorm due to the op having higher quantization sensitivity
for name, qmodule in quantsim_adascale.named_qmodules():
    if re.search(r'rmsnorm', qmodule.__class__.__name__.lower()):
        qmodule.param_quantizers['weight'] = QuantizeDequantize(shape=(), bitwidth=16, symmetric=False).to(next(quantsim_adascale.model.parameters()).device)

# ---
# #### 3.5 AdaScale

print("=" * 80)
print("3.5 AdaScale")
print("=" * 80)

from aimet_torch.experimental.adascale import apply_adascale

with event_marker("apply AdaScale", flush_ram=True):
    with place_model(quantsim_adascale.model, "cuda"):
        apply_adascale(qsim=quantsim_adascale,
                       data_loader=adascale_train_dataloader, #tqdm(adascale_train_dataloader, desc="AdaScale", total=adascale_iterations),
                       forward_fn=custom_forward,
                       num_iterations=adascale_iterations)

# ---
# ## 4. Evaluate and Export AdaScale Model

print("=" * 80)
print("4. Evaluate and Export AdaScale Model")
print("=" * 80)

# ---
# #### 4.1 AdaScale Evaluation

print("=" * 80)
print("4.1 AdaScale Evaluation")
print("=" * 80)

from aimet_torch.v2.utils import remove_activation_quantizers

if run_ppl_eval:
    with event_marker("AdaScale FP model eval"):
        with place_model(quantsim_adascale.model, torch.device('cuda')), remove_activation_quantizers(quantsim_adascale.model):
            adascaled_ppl = llm_evaluate_ppl_with_dataloader(model=quantsim_adascale.model, dataloader=wikitext_test_dataloader_adascale,
                                                              num_batches=num_eval_batches)
    print(f"PPL score of AdaScale model = {adascaled_ppl}")

# ---
# #### 4.2 Export AdaScale model

print("=" * 80)
print("4.2 Export AdaScale model")
print("=" * 80)

with event_marker("save AdaScale model", flush_ram=True):
    fp_ada_model = QuantizationSimModel.get_original_model(quantsim_adascale.model, qdq_weights=True)
    fp_ada_model.save_pretrained(adascale_dir)
    tokenizer_adascale.save_pretrained(adascale_dir)

# ---
# ### AdaScale Summary

print("=" * 80)
print("AdaScale Summary")
print("=" * 80)

from genai_lib.common.debug.profiler import EventProfiler
EventProfiler().report()
EventProfiler().json_dump(os.path.join(adascale_dir, 'profiling_stats.json'))

print(f"\nAdaScale model saved to: {adascale_dir}")

# ---
# ### Cleanup AdaScale Phase

print("=" * 80)
print("Cleanup AdaScale Phase")
print("=" * 80)

import gc

# Clean up AdaScale resources
del model_adascale
del quantsim_adascale
del fp_ada_model
del tokenizer_adascale
del wikitext_test_dataloader_adascale
del adascale_train_dataloader
del dummy_input_adascale

# Force garbage collection and clear CUDA cache
gc.collect()
torch.cuda.empty_cache()

print("AdaScale phase complete. Memory cleaned up.")

# ============================================================================
# PART 2: BASE QUANTIZATION PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: BASE QUANTIZATION PIPELINE")
print("=" * 80)

# Update model_id to point to AdaScale output
model_id = adascale_dir
print(f"Using AdaScale-optimized model from: {model_id}")

# ---
# ### 1.2 Setting NSP Target for Base

print("=" * 80)
print("1.2 Setting NSP Target for Base")
print("=" * 80)

# Android GEN4 and GEN5 is supported for this notebook
# Use the same PLATFORM_GEN as AdaScale phase
nsp_target = eval(f"NspTargets.{TARGET_PLATFORM}.GEN{PLATFORM_GEN}")

# Select quantsim config based on target
htp_config_file = f'htp_quantsim_config_{nsp_target.dsp_arch}.json'

onnx_dir = os.path.join(output_dir, 'base', 'onnx')
os.makedirs(onnx_dir, exist_ok=True)

import torch

#======================Configurable setting by users================================
from transformers import AutoConfig, AutoTokenizer
llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
# Set context length to be 2048, 4096, or 8192 here, user can change this value to ones' desire (but less than Phi4-mini' trained contex length)
context_length= 4096

# To help with debugging num_hidden_layers could be set to 2 to quickly verify the pipeline and export a two layer model for verification purposes
#llm_config.num_hidden_layers = 32
print(f'num_layer: {llm_config.num_hidden_layers}, context_length : {context_length},'
      f'num_hidden_size :{llm_config.num_attention_heads},  num_kv_heads: {llm_config.num_key_value_heads}')

#======================Fixed setting that should not be changed by users==============
# Auto-regression length: number of tokens to consume and number of logits to produce.
# This value should NOT be changed due to downstream consumption requirements

if 8192 == context_length:
    ARN = 7073
elif 4096 == context_length:
    ARN = 2073
elif 2048 == context_length:
    ARN = 1073
else:
    ARN = 573

setattr(llm_config, 'return_new_key_value_only', True)
setattr(llm_config, 'transposed_key_cache', True)
setattr(llm_config, 'use_combined_mask_input', True)
setattr(llm_config, 'use_position_embedding_input', True)
setattr(llm_config, '_attn_implementation', 'eager')
setattr(llm_config, '_attn_implementation_internal', 'eager')
setattr(llm_config, 'mask_neg', -7100) #-100
setattr(llm_config, 'partial_rotary_factor', 0.75)

llm_config.save_pretrained(output_dir)

# #### 2.2 Adapt FP32 model definition for inference on HTP
# The following adaptations are done to replace default attention module with attention definition that compatible with NSP backend
# - use conv instead of linear for Q,K,V,O projections
# - bypass attention and causal mask generation and replace with pre-generated 2D-mask input
# - output only newly created V and transposed K instead of entire augmented KV sequence
# - input pre-calculated positional embedding instead of position ids, thus bypass the embedding generation in the model

# In[ ]:


from transformers.models.phi3 import modeling_phi3
# from aimet_torch.pro.utils.profiler import event_marker
from genai_lib.common.debug.profiler import event_marker
with event_marker('FP model'):
    model = modeling_phi3.Phi3ForCausalLM.from_pretrained(model_id,cache_dir=cache_dir, config=llm_config)

    os.environ['TOKENIZERS_PARALLELISM'] = '0'
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
    ## Adjust the tokenizer to limit to context_length
    tokenizer.model_max_length = context_length
    tokenizer.save_pretrained(output_dir)

# In[ ]:


from transformers.models.phi3 import modeling_phi3
from transformers import cache_utils, modeling_attn_mask_utils
# from aimet_torch.pro.utils.profiler import event_marker
from llm_utils.qcphi4_adaptation import (
    QcPhi4Attention,
    bypass_update_causal_mask,
    bypass_Phi4RotaryEmbedding,
    Phi4MLP_prepare_conv,
    Phi4MLP_forward_conv,
    Phi4ForCausalLM_prepare_conv,
    Phi4ForCausalLM_forward,
    DynamicCache_update,
    DynamicCache_get_seq_length,
    update_attr
)

with event_marker("FP model adaptation configuration"):
    for layer in model.model.layers:
        layer.self_attn.__class__ = QcPhi4Attention
        layer.self_attn.num_key_value_heads = layer.self_attn.config.num_key_value_heads

    _original_qcphi4_attention_forward = QcPhi4Attention.forward
    _original_qcphi4_attention_forward_conv = QcPhi4Attention.forward_conv


    def qcphi4_attention_forward_451(self, *args, **kwargs):
        attn_output, attn_weights, _ = _original_qcphi4_attention_forward(
            self, *args, **kwargs
        )
        return attn_output, attn_weights


    def qcphi4_attention_forward_conv_451(self, *args, **kwargs):
        attn_output, attn_weights, _ = _original_qcphi4_attention_forward_conv(
            self, *args, **kwargs
        )
        return attn_output, attn_weights


    QcPhi4Attention.forward = qcphi4_attention_forward_451
    QcPhi4Attention.forward_conv = qcphi4_attention_forward_conv_451

    # Bypass attention_mask preparation
    assert update_attr(modeling_phi3.Phi3Model, '_update_causal_mask', bypass_update_causal_mask) or \
           update_attr(modeling_phi3.Phi3Model, '_prepare_decoder_attention_mask', bypass_update_causal_mask),  \
            f"neither _prepare_decoder_attention_mask(..) nor _update_causal_mask(..) found, Unknown Phi3Model definition {modeling_phi3.Phi3Model}"

    # Bypass rotary_emb module
    assert update_attr(modeling_phi3.Phi3RotaryEmbedding, 'forward', bypass_Phi4RotaryEmbedding), \
        f"Unknown RotaryEmbedding definition: {modeling_phi3.Phi3RotaryEmbedding}"

    # Adaptation to use Conv instead of linear
    setattr(modeling_phi3.Phi3MLP, 'prepare_conv', Phi4MLP_prepare_conv)
    setattr(modeling_phi3.Phi3MLP, 'forward_conv', Phi4MLP_forward_conv)
    setattr(modeling_phi3.Phi3ForCausalLM, 'prepare_conv', Phi4ForCausalLM_prepare_conv)
    update_attr(modeling_phi3.Phi3ForCausalLM, 'forward', Phi4ForCausalLM_forward)

    # Adapting KV$ management
    assert update_attr(cache_utils.DynamicCache, 'update', DynamicCache_update), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
    assert update_attr(cache_utils.DynamicCache, 'get_seq_length', DynamicCache_get_seq_length),  f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"

# Transformers >= 4.51 requires Phi3Model.forward() to receive a Cache object,
# while LLMForwardPassManager uses the legacy tuple KV-cache representation.
# Keep the compatibility conversion local to Part 2.
def part2_custom_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                         past_key_values=None, *args, **kwargs):
    if past_key_values is not None and not isinstance(past_key_values, cache_utils.Cache):
        past_key_values = cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    outputs = self.__original_forward__(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        *args,
        **kwargs,
    )

    # Part 2 normally uses return_dict=False. Preserve every returned value and
    # only convert the returned KV cache back to the legacy tuple representation.
    if isinstance(outputs, tuple):
        outputs = list(outputs)
        if len(outputs) > 1 and isinstance(outputs[1], cache_utils.Cache):
            outputs[1] = outputs[1].to_legacy_cache()
        return tuple(outputs)

    # Also support return_dict=True in case it is enabled later.
    if (
        getattr(outputs, "past_key_values", None) is not None
        and isinstance(outputs.past_key_values, cache_utils.Cache)
    ):
        outputs.past_key_values = outputs.past_key_values.to_legacy_cache()

    return outputs


# This is intentionally installed after Phi3ForCausalLM.forward has been adapted
# to Phi4ForCausalLM_forward, so __original_forward__ points to that adapted forward.
import types
model.__original_forward__ = model.forward
model.forward = types.MethodType(part2_custom_forward, model)


# ### 3. Complete the last step(s) of model adaptation
#
# The following model adaptation are enabled for inference:
# - apply linear to conv in attention, MLP and lmhead and arrange linear weights properly for conv

# In[ ]:


with event_marker('FP model adaptation for NSP backend completion'):
    for name, module in model.named_modules():
        if hasattr(module, "prepare_conv"):
            module.prepare_conv()


# ### 4. Model Evaluation

# In[ ]:


from torch.nn import CrossEntropyLoss
from llm_utils.forward_pass_wrapper import slice_inputs_and_run_successive_kvcache_inference

def ppl_eval(data_loader, forward_pass_manager, num_batches=0):
    if num_batches == 0:
        num_batches = len(data_loader)
    loss = 0

    if llm_config.num_hidden_layers < 10:
        num_batches = 1

    for batch_id, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Evaluating")):
        if batch_id >= num_batches:
            break
        outputs = slice_inputs_and_run_successive_kvcache_inference(forward_pass_manager, input_ids=batch['input_ids'])
        lm_logits = outputs["lm_logits"].cpu()

        # we can either pass input_ids or input_embeds in our fpm, hence with input_embeds we pass the labels.
        if 'input_ids' not in batch:
            batch['input_ids'] = batch['labels']

        lm_logits = lm_logits.reshape(batch['input_ids'].shape[0], -1, lm_logits.shape[-1])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous().to(shift_logits.device)
        loss_fct = CrossEntropyLoss()
        loss += loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    loss = loss / num_batches
    ppl = loss.exp()

    return ppl


# #### 4.1 FP32 PPL Eval

# In[ ]:


from llm_utils.forward_pass_wrapper import LLMForwardPassManager
orig_fpm = LLMForwardPassManager(cfg=llm_config,
                                 model=model,
                                 tokenizer=tokenizer,
                                 separate_tuple_input_output=False,
                                 num_tokens=ARN)

from llm_utils.wikitext_dataloader import get_wiki_dataset
train_dataloader, test_dataloader, _ = get_wiki_dataset(context_length, tokenizer, cache_dir)

if run_ppl_eval:
    with event_marker("FP eval"):
        with torch.no_grad():
            with orig_fpm.place_on_device("cuda"):
                orig_ppl = ppl_eval(test_dataloader, orig_fpm)

    print(f"ppl score of original fp model: {orig_ppl}")


# ### 5. Model Sample Input

# In[ ]:


from llm_utils.forward_pass_wrapper import get_position_embeddings_from_position_ids, prepare_combined_attention_mask, get_padded_kv_values, flatten_tensors

def get_dummy_data(config, tokenizer, device, separate_tuple_input_output, num_tokens=None, dtype=torch.float32):

    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    partial_rotary_factor = config.partial_rotary_factor

    max_tokens = tokenizer.model_max_length
    attention_mask = torch.ones((1, max_tokens), dtype=torch.long, device=device)

    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    position_ids = position_ids.clip(0, max_tokens - 1)
    position_ids = position_ids[..., :num_tokens]
    position_ids = position_ids.to(device=device)
    if config.use_combined_mask_input:
        past_kv_length = max_tokens - num_tokens
        attention_mask = prepare_combined_attention_mask(attention_mask, input_shape=(1, num_tokens),
                                                      past_key_values_length=past_kv_length, device=device,
                                                      mask_neg=llm_config.mask_neg, dtype=dtype)

    if config.use_position_embedding_input:
        position_ids = get_position_embeddings_from_position_ids(position_ids,
                                                                 head_dim=hidden_size/num_attention_heads,
                                                                 max_length=max_tokens,
                                                                 partial_rotary_factor=partial_rotary_factor,
                                                                 device=device, dtype=dtype,
                                                                 config=config)

    inputs = {
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'input_ids': torch.randint(0, len(tokenizer), (1, num_tokens), device=device)
    }

    inputs['past_key_values'] = get_padded_kv_values(past_size=max_tokens - num_tokens,
                                                     num_layers=num_layers,
                                                     hidden_size=hidden_size,
                                                     num_attention_heads=num_attention_heads,
                                                     num_kv_heads=num_kv_heads,
                                                     transposed_key_cache=config.transposed_key_cache,
                                                     device=device,
                                                     dtype=dtype)

    if separate_tuple_input_output:
        flattened_kvcache = tuple(flatten_tensors(inputs['past_key_values']))
        if isinstance(inputs['position_ids'], tuple):
            inputs = inputs['input_ids'], inputs['attention_mask'], inputs['position_ids'][0], inputs['position_ids'][1]
        else:
            inputs = inputs['input_ids'], inputs['attention_mask'], inputs['position_ids']
        inputs = inputs + flattened_kvcache

    return inputs


# ### 6. Prepare model using AIMET model preparer pro
#
# #### 6.1 KVCache MHA model preparation

# In[ ]:


import time
from aimet_torch import onnx_utils

from genai_lib.llm.model_preparation_utils import llm_build_preparer_converter_args

from qti.aisw.preparer_api import model_preparer
from qti.aisw.emitter.utils.torch_utils import load_torch_model_using_safetensors

# Setting this flag to False means that the prepared model will be flattened
onnx_utils.EXPORT_TO_ONNX_DIRECT = True

def _get_past_key_values_names(sfx, n_layers):
    all_kvs = []
    for i in range(n_layers):
        all_kvs.append(f'past_key_{i}_{sfx}')
        all_kvs.append(f'past_value_{i}_{sfx}')
    return all_kvs

dummy_input = get_dummy_data(llm_config,
                             tokenizer, 'cpu', separate_tuple_input_output=False, num_tokens=ARN, dtype=model.dtype)
input_names = ['input_ids', 'attention_mask']
input_names += ['position_ids_cos', 'position_ids_sin'] if llm_config.use_position_embedding_input else ['position_ids']
input_names += _get_past_key_values_names('in', llm_config.num_hidden_layers)
output_names = ['logits'] +  _get_past_key_values_names('out', llm_config.num_hidden_layers)

# Build converter args
converter_args = llm_build_preparer_converter_args(llm_config.num_hidden_layers, input_names, use_qairt_mpp=True) # Build converter args

prepare_path = os.path.join(output_dir, 'prepare')
os.makedirs(prepare_path, exist_ok=True)
prepare_filename = f'{model_name}_kvcache_{llm_config.num_hidden_layers}_layer'

skip_prepare = False
if skip_prepare:
    with event_marker(f"KVCache load pre-prepared {prepare_filename}", flush_ram=True):
        prepared_model_path = os.path.join(prepare_path, f'{prepare_filename}.py')
        if not os.path.exists(prepared_model_path):
            raise ValueError(f"prepared artifacts not found in {prepare_path}")
        else:
            print(f'WARNING: preparation skipped for model={prepare_filename}, prepared at {time.ctime(os.path.getmtime(prepared_model_path))}')
            prepared_model = load_torch_model_using_safetensors(path=prepare_path, filename=prepare_filename, model_name=prepare_filename)
else:
    with event_marker("KVCache prepare model", flush_ram=True):
        model.num_logits_to_return = ARN # configuring the model for KVCache mode
        prepared_model = model_preparer.prepare_model(model,
                              dummy_input,
                              model_name = prepare_filename,
                              filename = prepare_filename,
                              path = prepare_path,
                              input_names = input_names,
                              output_names = output_names,
                              onnx_export_args = {"opset_version":17},
                              converter_args = converter_args,
                              keep_original_model_structure = False, # Flatten the model to enable weight-sharing by setting
                              skipped_optimizers = ['eliminate_common_subexpression',
                                                   'eliminate_nop_with_unit',
                                                   'eliminate_duplicate_initializer'
                                                   ],
                               return_prepare_model = False
                            )
        prepared_model = load_torch_model_using_safetensors(path=prepare_path, filename=prepare_filename, model_name=prepare_filename)



# Calculate ppl score for prepared fp model
fp_prepared_fpm = LLMForwardPassManager(cfg=llm_config,
                                        model=prepared_model,
                                        tokenizer=tokenizer,
                                        separate_tuple_input_output=True,
                                        num_tokens=ARN)

if run_ppl_eval:
    with event_marker("KVcache prepared FP eval", flush_ram=True):
        with torch.no_grad():
            with fp_prepared_fpm.place_on_device("cuda"):
                prepared_kvcache_ppl = ppl_eval(test_dataloader, fp_prepared_fpm)

    # This should be very close (<1e-4 delta) to original model's perplexity
    # If the perplexity score goes further up, it indicates the AIMET/QNN pair is producing a faulty prepared model
    print(f"ppl score of KVCACHE prepared fp model: {prepared_kvcache_ppl}\n"
          f"orig ppl - prepared ppl = {orig_ppl - prepared_kvcache_ppl}")


# #### 7.1 Create quantsim configured for QNN HTP target
#
# The following member function allows creation of a shallow copied model. This shallow copied model is a separate model object from the original, but contains shared weights, biases, and parameters. As a result, the shallow copied model has very little memory overhead, which is useful for PTQ techniques like sequential MSE that expect separate FP and QuantSim models.

# In[ ]:


# Helper function that creates a shallow copy of the provided model
# Creates a new model object, but all the underlying parameters are shared
import copy
from copy import deepcopy
import functools

def copy_model_with_shared_weights(source_model):
    target_model = deepcopy(source_model)
    for name, source_parameter in source_model.named_parameters():
        pre, _, post = name.rpartition('.')
        pre_obj = functools.reduce(getattr, [target_model] + pre.split('.')) if pre else target_model
        setattr(pre_obj, post, source_parameter)
    return target_model


# In[ ]:


from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel

sim_fpm = LLMForwardPassManager(cfg=llm_config,
                                model=copy_model_with_shared_weights(prepared_model), # to avoid creating the sim in_place on the original model
                                tokenizer=tokenizer,
                                separate_tuple_input_output=True,
                                num_tokens=ARN)

dummy_input = get_dummy_data(llm_config,
                             tokenizer, 'cuda', separate_tuple_input_output=True,
                             num_tokens=ARN, dtype=sim_fpm.dtype)

with event_marker("create KVCache Quantsim"):
    with sim_fpm.place_on_device("cuda"):
        quantsim = QuantizationSimModel(model=sim_fpm.model,
                                        quant_scheme=QuantScheme.post_training_tf,
                                        dummy_input=dummy_input,
                                        default_output_bw=16,
                                        default_param_bw=4,
                                        in_place=True,
                                        config_file=htp_config_file)


# #### 7.2 Setting 16bit x 8bit matmuls
#
# To keep key and value tensors as 8 bits, reducing data I/O costs associated with KV-cache orchestration.

# In[ ]:


from aimet_torch.v2.experimental.quantsim_utils import set_matmul_second_input_producer_to_8bit_symmetric

set_matmul_second_input_producer_to_8bit_symmetric(quantsim)


# #### 7.3 Concat encoding unification
#
# Configuring concat ops to have shared encoding on input and output activations.

# In[ ]:


from aimet_torch.v2.experimental import propagate_output_encodings
import aimet_torch.elementwise_ops as aimet_ops

propagate_output_encodings(quantsim, aimet_ops.Concat)


# #### 7.4 Manual Mixed Precision
#
# Applying mixed precision configuration to ops

# In[ ]:


from llm_utils.mixed_precision_overrides import ManualQuantsimMixedPrecisionConfig

config_file = "./config/mixed_precision_config/exceptions.json"

quantsim_adjuster = ManualQuantsimMixedPrecisionConfig(mixed_precision_config_file=config_file)
quantsim_adjuster.apply_exceptions(quantsim)


# In[ ]:


from aimet_torch.v2.nn.modules.custom import QuantizedRmsNorm
from aimet_torch.v2.quantization.affine import QuantizeDequantize

# Make RMSNorm encodings per-tensor (they default to per-channel)
for name, qmodule in quantsim.named_qmodules():
    if isinstance(qmodule, QuantizedRmsNorm):
        qmodule.param_quantizers['weight'] = QuantizeDequantize(shape=(), bitwidth=16, symmetric=False).to(qmodule.weight.device)


# #### 7.6 Calibration
#
# Compute activation encodings using AIMET

# In[ ]:


def _forward_fn(model, kwargs):
    data_loader = kwargs['data_loader']
    fpm = kwargs['fpm']
    max_iterations = kwargs['num_batches']
    for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
        if batch_id < max_iterations:
             slice_inputs_and_run_successive_kvcache_inference(fpm, input_ids=batch['input_ids'])
        else:
            break
kwargs = {
   'data_loader': train_dataloader,
    'fpm': sim_fpm,
    'num_batches': 100
}

with event_marker("compute encoding", flush_ram=True):
     with sim_fpm.place_on_device("cuda"):
        quantsim.compute_encodings(_forward_fn, kwargs)


# #### 7.7 Eval KV Cache sim model

# In[ ]:

if run_ppl_eval:
    with event_marker("KV cache sim eval", flush_ram=True):
        with torch.no_grad():
            with sim_fpm.place_on_device("cuda"):
                sim_ppl = ppl_eval(test_dataloader, sim_fpm)

    print(f"ppl score of KVCACHE sim fp model: {sim_ppl}\n"
          f"orig ppl - kvcache sim ppl = {orig_ppl - sim_ppl}")


# ### 8. Export
#
# The pipeline call below would export onnx model, encoding and test vector for KVCache models.
#
# #### 8.1 Export KVCache Model

# In[ ]:


from aimet_torch.utils import change_tensor_device_placement
from aimet_torch.onnx_utils import OnnxExportApiArgs

onnx_api_args = OnnxExportApiArgs(input_names=input_names,output_names=output_names)
sample_inputs = change_tensor_device_placement(dummy_input, torch.device('cpu'))
with event_marker("KVCache export", flush_ram=True):
    quantsim.export(onnx_dir, model_name, sample_inputs, onnx_export_args=onnx_api_args)

# Export chat template
if getattr(tokenizer, "chat_template", None):
    with open(os.path.join(output_dir, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(tokenizer.chat_template)
else:
    print("No chat_template found on tokenizer; nothing to export.")

# Export generation config
model.generation_config.save_pretrained(output_dir)

# #### 8.2 Generating test vectors for QNN SDK

# In[ ]:


from llm_utils.test_vectors import generate_test_vectors

test_vector_layers = [
  "rms_norm_\\d+",
  "lm_head_conv_Conv$"
]

with event_marker("generate test vector"):
    with sim_fpm.place_on_device("cuda"):
        generate_test_vectors(quantsim, sim_fpm, train_dataloader, output_dir, num_batches=1, test_vector_layers=test_vector_layers, input_names=input_names)


# ### Summary

# In[ ]:


# from aimet_torch.pro.utils.profiler import EventProfiler
from genai_lib.common.debug.profiler import EventProfiler
EventProfiler().report()
EventProfiler().json_dump(os.path.join(output_dir, 'profiling_stats.json'))

if run_ppl_eval:
    import json
    with open(f'{output_dir}/ppl.json', 'wt') as f:
        json.dump({
            "original": float(orig_ppl),
            "prepared_kvcache": float(prepared_kvcache_ppl),
            "QuantSim": float(sim_ppl),
        }, f, indent=2)


print("=" * 80)
print("Script Complete!")
print("=" * 80)
print(f"Final quantized model exported to: {onnx_dir}")
