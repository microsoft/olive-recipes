'''
SPINQUANT QUANTIZATION
'''

if __name__ != '__main__':
    raise Exception("Killing multiprocessing spawn started by Converter during model preparation.")




try:
    # Required for proper Python environment configuration of qairt-dev
    import qairt  # noqa: F401  # pylint: disable=unused-import
except ImportError as exc:
    raise ImportError(
        "Failed to import QAIRT SDK - please install olive-ai[qairt] to use QAIRT passes."
        "If already installed, please run `qairt-vm -i` for help troubleshooting issues."
    ) from exc


# ---
# ### Configuration Loading System
# Supports loading configuration from JSON file with 3-tier priority:
# 1. JSON config file (if provided)
# 2. Environment variables
# 3. Default values

import json
import argparse
import os

# Parse command-line arguments for optional config file
parser = argparse.ArgumentParser(
    description="Llama 3.1 8B Instruct AdaScale + Quantization Script",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Configurable Variables (via JSON config file, environment variables, or defaults):
Spinquant Phase:
  Model:
    SPINQUANT_MODEL_ID            HuggingFace model ID or local path (default: Qwen/Qwen3-8B)
    SPINQUANT_CACHE_DIR           Cache directory for downloaded model files (default: ./cache_dir)
    SPINQUANT_NUM_HIDDEN_LAYERS   Number of hidden layers, 0=use model default (default: 0)

  SpinQuant:
    SPINQUANT_ENABLED             Run SpinQuant rotation (default: True)
    SPINQUANT_CONTEXT_LENGTH      Context length for tokenizer and dataloaders (default: 4096)
    SPINQUANT_TORCH_DTYPE         Torch dtype for model loading e.g. torch.float32 (default: torch.float32)

  Evaluation:
    SPINQUANT_RUN_PPL_EVAL        Run perplexity evaluation (default: True)
    SPINQUANT_NUM_EVAL_BATCHES    Number of batches for PPL eval, 0=all (default: 0)
    SPINQUANT_WIKI_DATASET_PATH   Local path to wikitext dataset, downloads if not set (default: None)
    SPINQUANT_ADASCALE_DATASET    Calibration dataset: C4 or SHAREGPT4O (default: C4)

      Adapters (LoRA, optional):
    SPINQUANT_ADAPTER_NAMES       Comma-separated adapter names (default: none)
    SPINQUANT_ADAPTER_PATHS       Comma-separated adapter paths, must match ADAPTER_NAMES (default: none)

    SPINQUANT_TARGET_PLATFORM     Target platform: Windows or Android (default: Windows)

AdaScale Phase:
  MODEL_ID                      Path to the initial qwen 3 8B model
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
  BASE_CONTEXT_LENGTH           Context length for base model (default: 4173)
  ARN                           Auto-regression length (default: 2073)
  ENABLE_RIGHT_PADDING          Enable right padding of kvcache (default: True)
  APPLY_DECODER_SEQMSE          Apply SeqMSE to decoder (default: False)
  APPLY_LM_HEAD_SEQMSE          Apply SeqMSE to LM head (default: False)
  APPLY_DECODER_LPBQ            Apply LPBQ to decoder (default: False)
  APPLY_LM_HEAD_LPBQ            Apply LPBQ to LM head (default: True)
  ACTIVATION_CLIPPING_CLAMP_VAL Activation clipping value (default: None)
  EMBEDDING_TABLE_BITWIDTH      Embedding table bitwidth: 8 or 16 (default: 8)
  ENABLE_FP16                   Enable FP16 flow (default: False)
  SKIP_PREPARE                  Skip model preparation (default: False)
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
  python qwen3.py --config my_config.json
  python qwen3.py --help
""",
)
parser.add_argument(
    "--config", type=str, default=None, help="Path to JSON configuration file"
)
args, unknown = parser.parse_known_args()

# Load JSON config if provided
json_config = {}
if args.config:
    try:
        with open(args.config, "r") as f:
            json_config = json.load(f)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Warning: Config file not found: {args.config}")
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in config file: {e}")





def get_config_value(key, default, value_type="str"):
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
        if value_type == "bool":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "t", "yes")
        elif value_type == "int":
            return int(value)
        elif value_type == "none":
            return value
        else:  # str
            return str(value) if value is not None else None

    # Priority 2: Check environment variable
    env_value = os.getenv(key)
    if env_value is not None:
        if value_type == "bool":
            return env_value.lower() in ("true", "1", "t")
        elif value_type == "int":
            return int(env_value)
        elif value_type == "none":
            return env_value
        else:  # str
            return env_value

    # Priority 3: Use default value
    return default





import os
import torch



print("=" * 80)
print("PART 1: Spinquant Optimziation")
print("=" * 80)

print("=" * 80)
print("PART 1.1: Spinquant feature knobs")
print("=" * 80)


# Feature knobs
context_length = get_config_value("SPINQUANT_CONTEXT_LENGTH", 4096, "int")
enable_spinquant = get_config_value("SPINQUANT_ENABLED", True, "bool")

# Speed knobs
run_ppl_eval = get_config_value("SPINQUANT_RUN_PPL_EVAL", True, "bool")
torch_dtype = eval(get_config_value("SPINQUANT_TORCH_DTYPE", "torch.float32"))
num_eval_batches = get_config_value("SPINQUANT_NUM_EVAL_BATCHES", 0, "int")

assert enable_spinquant, "ENABLE_SPINQUANT must be True"


print("=" * 80)
print("PART 1.2: Setting NSP target")
print("=" * 80)


import sys
from utilities.nsptargets import NspTargets

# setup Target platform and its generation
TARGET_PLATFORM = get_config_value("SPINQUANT_TARGET_PLATFORM", "Windows").capitalize()

# Android GEN4 and GEN5 is supported for this notebook
PLATFORM_GEN = get_config_value("PLATFORM_GEN", 3, "int")

nsp_target = eval(f"NspTargets.{TARGET_PLATFORM}.GEN{PLATFORM_GEN}")

# Select quantsim config based on target
htp_config_file = f'htp_{nsp_target.dsp_arch}'
print(htp_config_file)


print("=" * 80)
print("PART 2 Instantiate and eval hf model")
print("=" * 80)

import aimet_torch
aimet_torch.quantization.set_backend("torch_builtins")
from aimet_torch.utils import place_model
from copy import deepcopy
from genai_lib.common.debug.profiler import event_marker
from genai_lib.common.debug.recipe_logger import llm_lib_log_env_info, recipe_dump_init, llm_lib_log_property, Property, llm_lib_log_metric, ModelType, Metric

model_id = get_config_value("SPINQUANT_MODEL_ID", "Qwen/Qwen3-8B")
cache_dir = get_config_value("CACHE_DIR", './cache_dir')
output_dir = get_config_value("OUTPUT_DIR", "./output_dir" )
spinquant_output_dir = os.path.join(output_dir, 'spinquant')
os.makedirs(spinquant_output_dir, exist_ok=True)

adapter_names = get_config_value("SPINQUANT_ADAPTER_NAMES", "").split(',') if get_config_value("SPINQUANT_ADAPTER_NAMES", "") else []
adapter_paths = get_config_value("SPINQUANT_ADAPTER_PATHS", "").split(',') if get_config_value("SPINQUANT_ADAPTER_PATHS", "") else []
assert len(adapter_names) == len(adapter_paths), f"Got {len(adapter_names)} adapter names but {len(adapter_paths)} adapter paths"

recipe_dump_init(spinquant_output_dir, "genai_lib_debug")
llm_lib_log_env_info()
llm_lib_log_property({Property.context_length : context_length})


print("=" * 80)
print("PART 2.1 Load HF Model")
print("=" * 80)

from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoConfig, AutoProcessor

model_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
is_VLM = hasattr(model_config, "vision_config")

num_hidden_layers = get_config_value("SPINQUANT_NUM_HIDDEN_LAYERS", 0, "int")
if is_VLM:
    model_config.text_config.num_hidden_layers = num_hidden_layers if num_hidden_layers > 0 else model_config.text_config.num_hidden_layers
else:
    model_config.num_hidden_layers = num_hidden_layers if num_hidden_layers > 0 else model_config.num_hidden_layers
llm_config = getattr(model_config, "text_config", model_config)

with event_marker('HuggingFace FP model creation'):
    os.environ['TOKENIZERS_PARALLELISM'] = '0'
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)

    if is_VLM:
        model = AutoModelForVision2Seq.from_pretrained(model_id, config=model_config, torch_dtype=torch_dtype)
        processor.tokenizer.model_max_length = context_length
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, torch_dtype=torch_dtype, cache_dir=cache_dir)
        processor.model_max_length = context_length


print("=" * 80)
print("PART 2.2 Instantiate dataloaders")
print("=" * 80)

from llm_utils.wikitext_dataloader import get_wiki_dataset
from llm_utils.generic_dataloader import get_local_dataset
from llm_utils.sharegpt4o_dataloader import get_sharegpt4o_dataset
from llm_utils.agentic_dataloader import get_agentic_dataset

with event_marker("Instantiate wikitext dataloader"):
    _, wikitext_test_dataloader, _ = get_wiki_dataset(context_length, processor.tokenizer if is_VLM else processor, cache_dir, path = get_config_value('SPINQUANT_WIKI_DATASET_PATH', None, "none"))

adascale_dataset = get_config_value("SPINQUANT_ADASCALE_DATASET", "SHAREGPT4O" if is_VLM else "C4")


print("=" * 80)
print("PART 2.2 Eval HF model")
print("=" * 80)

from genai_lib.llm.evaluation_utils import llm_evaluate_ppl_with_dataloader
from peft import PeftModel

if run_ppl_eval:
    with event_marker("HuggingFace FP model eval"):
        with place_model(model, torch.device('cuda')):
            orig_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wikitext_test_dataloader, num_batches=num_eval_batches)
    llm_lib_log_metric(ModelType.hf_model, Metric.ppl, orig_ppl)
    print(f"PPL score of HuggingFace FP model = {orig_ppl}")

    for adapter_name, adapter_path in zip(adapter_names, adapter_paths):
        peft_model = PeftModel.from_pretrained(deepcopy(model), adapter_path)
        with event_marker(f"{adapter_name} - SpinQuant evaluation"):
            with place_model(peft_model, torch.device('cuda')):
                orig_peft_ppl = llm_evaluate_ppl_with_dataloader(model=peft_model, dataloader=wikitext_test_dataloader, num_batches=num_eval_batches)
        llm_lib_log_metric(ModelType.hf_model, Metric.ppl, orig_peft_ppl)
        print(f"{adapter_name} - PPL score of HuggingFace FP model = {orig_peft_ppl}")
        del peft_model




print("=" * 80)
print("PART 3 Spinquant")
print("=" * 80)

from aimet_torch.experimental.spinquant.spinquant_optimizer import SpinQuant
from llm_utils.spinquant_adascale_utils import apply_spinquant_r1_to_adapter, capture_norm_fusion_factors, apply_norm_scaling_to_lora
from safetensors.torch import load_file, save_file
from shutil import copytree, ignore_patterns

if enable_spinquant:
    model.to(torch.float32)
    # Capture weight and scale of norm ops, before SpinQuant fold the norms, so that we can also fold norm weight/scale on adapters
    factors_llm = capture_norm_fusion_factors(model,
                                              layers_iterable = model.model.language_model.layers if is_VLM else model.model.layers,
                                              input_ln_attr = 'input_layernorm',
                                              post_ln_attr = 'post_attention_layernorm',
                                              final_norm_attr = 'language_model.norm' if is_VLM else 'model.norm',)

    with event_marker("Apply Rotations to Model"):
        # Manually untie embedding weights after init, as 'tie_word_embeddings=False' on config during init results in LM-Head initialized with random weights
        model.lm_head = deepcopy(model.lm_head)
        model.tie_word_embeddings = False
        spinquant = SpinQuant()
        spinquant.apply_spinquant(model)

    if run_ppl_eval:
        with event_marker("SpinQuant QuantSim evaluation"):
            with place_model(model, torch.device('cuda')):
                spinquant_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wikitext_test_dataloader, num_batches=num_eval_batches)
        llm_lib_log_metric(ModelType.adapted_model, Metric.ppl, spinquant_ppl)
        print(f"PPL score of SpinQuant model = {spinquant_ppl}")

    adapter_paths_spinquant = []
    for adapter_name, adapter_path in zip(adapter_names, adapter_paths):

        with event_marker(f"{adapter_name} - Apply Rotations to Adapter"):
            adapter_weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))

            # SpinQuant requires first forward-folding the norm ops onto layers with left-hand-side rotations
            # AIMET handles this for main model, but we must manually apply norm weight/scale to lora_A
            adapter_weights = apply_norm_scaling_to_lora(peft_state_dict = adapter_weights,
                                                         factors = factors_llm,
                                                         layername_filter = "lang" if is_VLM else "",
                                                         norm_to_targets={'input_layernorm': ('q_proj','k_proj','v_proj','qkv'),
                                                                          'post_attention_layernorm': ('gate_proj','up_proj'),},
                                                         layer_index_patterns = (r"layers\.(\d+)\.",),
                                                         strict=True)

            adapter_weights = apply_spinquant_r1_to_adapter(peft_state_dict = adapter_weights,
                                                            language_hidden_size = model_config.text_config.hidden_size if is_VLM else model_config.hidden_size,
                                                            vision_hidden_size = model_config.vision_config.hidden_size if is_VLM else None,
                                                            language_layers_filter = "lang" if is_VLM else "",
                                                            vision_layers_filter = "vis" if is_VLM else "",
                                                            layers_with_lhs_rotations = ("q_proj", "k_proj", "v_proj", "qkv", "gate_proj", "up_proj"),
                                                            layers_with_rhs_rotations = ("o_proj", "down_proj"))

            # Save the SpinQuant'ed version of the adapters
            adapter_path_spinquant = os.path.join(spinquant_output_dir, f"{adapter_name}_spinquant")
            adapter_paths_spinquant.append(adapter_path_spinquant)
            os.makedirs(adapter_path_spinquant, exist_ok=True)
            copytree(src=adapter_path,
                     dst=adapter_path_spinquant,
                     dirs_exist_ok=True,
                     ignore=ignore_patterns("adapter_model.safetensors"))
            save_file(adapter_weights, os.path.join(adapter_path_spinquant, "adapter_model.safetensors"))

        if run_ppl_eval:
            peft_model = PeftModel.from_pretrained(deepcopy(model), adapter_path_spinquant)
            with event_marker(f"{adapter_name} - SpinQuant evaluation"):
                with place_model(peft_model, torch.device('cuda')):
                    spinquant_peft_ppl = llm_evaluate_ppl_with_dataloader(model=peft_model, dataloader=wikitext_test_dataloader, num_batches=num_eval_batches)
            llm_lib_log_metric(ModelType.adapted_model, Metric.ppl, spinquant_peft_ppl)
            print(f"{adapter_name} - PPL score of SpinQuant model = {spinquant_peft_ppl}")
            del peft_model

    # Ensure AdaScale uses rotated adapters
    adapter_paths = adapter_paths_spinquant
    model.to(torch_dtype)



print("=" * 80)
print("PART 4 Export Model")
print("=" * 80)


print("=" * 80)
print("PART 4.1 Register model config for adascale")
print("=" * 80)


from transformers.models.qwen3 import modeling_qwen3
from aimet_torch.experimental.adascale import adascale_optimizer

adascale_optimizer.adascale_model_config_dict[modeling_qwen3.Qwen3Model] = adascale_optimizer.AdaScaleModelConfig(
        block_type=modeling_qwen3.Qwen3DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    )


print("=" * 80)
print("PART 4.2 Redeine forward for JIT tracing in Quantsim creation")
print("=" * 80)


from transformers import PreTrainedModel, DynamicCache
from dataclasses import is_dataclass

class ONNXExportableModuleWithCache(torch.nn.Module):
    """
    Helper class to enable Torch JIT trace and ONNX export of HuggingFace models that produce and consume Cache objects
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the wrapped HF model when not found on this wrapper.
        This is called only when normal lookup on this module fails.
        """
        try:
            # Let nn.Module's own __getattr__ try first (parameters/buffers/etc.)
            return super().__getattr__(name)
        except AttributeError:
            # Fallback to the underlying model
            return getattr(self.model, name)

    # pylint: disable=keyword-arg-before-vararg
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        *args,
        **kwargs
    ):
        """Redefine model forward to convert to/from Huggingface DynamicCache objects"""
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        kwargs.pop("return_dict", None)
        kwargs.pop("num_logits_to_return", None)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            num_logits_to_return=0,
            return_dict=True, # newer transformers deprecated support for return_dict False
            *args,
            **kwargs
        )

        if is_dataclass(output):
            output = dict(output)
            lm_logits = output.pop('logits')
            new_past_key_values = output.pop('past_key_values')
            if len(output.keys()) > 0:
                print(f"output dict has extra keys = {list(output.keys())}")
            rest = tuple(list(output.values()))
        else:
            lm_logits, new_past_key_values, *rest = output
        if len(rest)>0:
            print(f"output dict has extra tensors = {rest}")
        if isinstance(new_past_key_values, DynamicCache):
            new_past_key_values = new_past_key_values.to_legacy_cache()
        return lm_logits, new_past_key_values, *rest

model = ONNXExportableModuleWithCache(model)


print("=" * 80)
print("PART 4.3 Export Model")
print("=" * 80)
with event_marker("Save model with SpinQuant weights", flush_ram=True):
    #fp_qdq_model = QuantizationSimModel.get_original_model(quantsim.model, qdq_weights=True) if enable_adascale else model
    model.model.save_pretrained(spinquant_output_dir)
    processor.save_pretrained(spinquant_output_dir)


from genai_lib.common.debug.profiler import EventProfiler
EventProfiler().report()
EventProfiler().json_dump(os.path.join(spinquant_output_dir, 'profiling_stats.json'))






'''
ADASCALE OPTIM
'''


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
                print(
                    f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                    end="",
                )

        urllib.request.urlretrieve(c4_url, c4_gz_file, reporthook=download_progress)
        print("\nDownload complete!")

        # Decompress the file
        print(f"Decompressing {c4_filename}.gz...")
        with gzip.open(c4_gz_file, "rb") as f_in:
            with open(c4_file, "wb") as f_out:
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


print("=" * 80)
print("PART 1.1: Notebook configs")
print("=" * 80)


import os
# Feature knobs
context_length = get_config_value("ADASCALE_CONTEXT_LENGTH", 2048, "int")

# Quantization knobs
adascale_iterations = get_config_value("ADASCALE_ITERATIONS", 5000, "int")
# Speed knobs
run_ppl_eval = get_config_value("ADASCALE_RUN_PPL_EVAL", True, "bool")
enable_bf16 = get_config_value("ADASCALE_ENABLE_BF16", False, "bool")
num_eval_batches = get_config_value("ADASCALE_NUM_EVAL_BATCHES", 0, "int")


print("=" * 80)
print("PART 1.2: Setting NSP Target")
print("=" * 80)

import sys
from utilities.nsptargets import NspTargets

os.environ['HF_HOME']="./"
# setup Target platform and its generation
TARGET_PLATFORM = get_config_value("ADASCALE_TARGET_PLATFORM", "Windows").capitalize()

# Android GEN4 and GEN5 is supported for this notebook
PLATFORM_GEN = get_config_value("PLATFORM_GEN", 3, "int")

nsp_target = eval(f"NspTargets.{TARGET_PLATFORM}.GEN{PLATFORM_GEN}")

# Select quantsim config based on target
htp_config_file = "htp_quantsim_config_v81.json"
print(htp_config_file)


print("=" * 80)
print("PART 2: Instantiate and eval hf model")
print("=" * 80)


import torch
from transformers import AutoModelForCausalLM
from aimet_torch.utils import place_model, change_tensor_device_placement
# from genai_lib.common.debug.profiler import event_marker

model_name = get_config_value("ADASCALE_MODEL_NAME", 'adascaled_model')
model_id = get_config_value("ADASCALE_MODEL_ID", spinquant_output_dir)
cache_dir = get_config_value("CACHE_DIR", './cache_dir')
output_dir = get_config_value("OUTPUT_DIR", "./output_dir")
adascale_dir = os.path.join(output_dir, 'adascale_output')
os.makedirs(adascale_dir, exist_ok=True)

# Recipe_logger: Initialize the logger and log environment details
# Note: This cell (and the corresponding cells with Recipe_logger tag) can be removed after dumping and verifying the recipe without impacting notebook functionality
from genai_lib.common.debug.recipe_logger import llm_lib_log_env_info, recipe_dump_init

recipe_dump_init(adascale_dir, "genai_lib_debug")

llm_lib_log_env_info()


print("=" * 80)
print("PART 2.1: Load hf model")
print("=" * 80)


from transformers import AutoConfig, AutoTokenizer
llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
num_hidden_layers = get_config_value("ADASCALE_NUM_HIDDEN_LAYERS", 0, "int")
llm_config.num_hidden_layers = num_hidden_layers if num_hidden_layers > 0 else llm_config.num_hidden_layers
print(f'num_layer: {llm_config.num_hidden_layers}, context_length: {context_length}, '
      f'num_hidden_size: {llm_config.num_attention_heads}, num_kv_heads: {llm_config.num_key_value_heads}')

model = AutoModelForCausalLM.from_pretrained(model_id, config=llm_config, cache_dir=cache_dir, torch_dtype = torch.bfloat16 if enable_bf16 else torch.float32)

os.environ['TOKENIZERS_PARALLELISM'] = '0'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
# Adjust the tokenizer to limit to context_length
tokenizer.model_max_length = context_length


print("=" * 80)
print("PART 2.2: Instantiate dataloaders")
print("=" * 80)

from llm_utils.wikitext_dataloader import get_wiki_dataset
from llm_utils.generic_dataloader import get_local_dataset


_, wikitext_test_dataloader, _ = get_wiki_dataset(context_length, tokenizer, cache_dir) #path = os.getenv('WIKI_DATASET_PATH', "/prj/corp/airesearch/morpheus/lasvegas/chipsets/common/datasets/wikitext/local_wiki"))


C4_DATASET_PATH = download_c4_dataset_if_needed(cache_dir)

adascale_train_dataloader, _ = get_local_dataset(context_length, tokenizer, json_path = C4_DATASET_PATH, key = "input_ids",
                                                     batch_size = get_config_value("ADASCALE_BATCH_SIZE", 2, "int"),
                                                     percent_dataset_to_load = get_config_value("ADASCALE_PERCENT_DATASET_TO_LOAD", 1, "int"),
                                                     num_samples = get_config_value("ADASCALE_NUM_SAMPLES", 500, "int"))



print("=" * 80)
print("PART 2.3: Eval HF Model")
print("=" * 80)

from genai_lib.llm.evaluation_utils import llm_evaluate_ppl_with_dataloader

if run_ppl_eval:
    with place_model(model, torch.device('cuda')):
        orig_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wikitext_test_dataloader, num_batches=num_eval_batches)

    print(f"PPL score of HuggingFace FP model = {orig_ppl}")

from genai_lib.common.debug.recipe_logger import llm_lib_log_property, Property
from genai_lib.common.debug.recipe_logger import llm_lib_log_metric, ModelType, Metric

# Recipe_logger: Log the context_length property and the metrics.

llm_lib_log_property({Property.context_length : context_length})

if run_ppl_eval:
    llm_lib_log_metric(ModelType.hf_model, Metric.ppl, orig_ppl)


print("=" * 80)
print("PART 3: Adascale")
print("=" * 80)


print("=" * 80)
print("PART 3.1: Redefine forward for JIT tracing in Quantsim Creation")
print("=" * 80)


import torch
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
model.__original_forward__ = model.forward
# Replace with custom forward
model.forward = types.MethodType(custom_forward, model)


print("=" * 80)
print("PART 3.2: Create quantsim configured for QNN HTP target")
print("=" * 80)

from aimet_torch.v2.quantsim import QuantizationSimModel

dummy_input = torch.randint(0, 1, (1, 1, context_length), device="cuda")

# with event_marker("create KVCache Quantsim"):
with place_model(model, "cuda"):
    model.config.return_dict=False
    quantsim = QuantizationSimModel(model=model,
                                    quant_scheme=0,
                                    # quant_scheme=QuantScheme.post_training_tf,
                                    default_output_bw=16,
                                    default_param_bw=4,
                                    in_place=True,
                                    dummy_input=tuple(list(dummy_input)),
                                        config_file=htp_config_file)


from aimet_torch.v2.experimental import propagate_output_encodings
from aimet_torch.nn.modules import custom as aimet_ops

propagate_output_encodings(quantsim, aimet_ops.Concat)

print("=" * 80)
print("PART 3.3: Enable per channel quantization")
print("=" * 80)

from aimet_torch.v2.nn.true_quant import QuantizedLinear
from aimet_torch.v2.quantization.affine import QuantizeDequantize

for name, qmodule in quantsim.named_qmodules():
    if isinstance(qmodule, QuantizedLinear):
        assert (len(qmodule.weight.shape)) == 2, f"Per channel quantization for linear weights is only supported for 2d weights, got shape: {qmodule.weight.shape} instead"
        qmodule.param_quantizers["weight"] = QuantizeDequantize(shape=(qmodule.weight.shape[0], 1),
                                                                bitwidth=qmodule.param_quantizers["weight"].bitwidth,
                                                                symmetric=qmodule.param_quantizers["weight"].symmetric).to(next(quantsim.model.parameters()).device)


print("=" * 80)
print("PART 3.4: Manual mixed precision + disable unneeded quantizers")
print("=" * 80)



import re

# Remove quantizers for non decoder blocks
quantsim.model.model.embed_tokens.param_quantizers["weight"] = None
quantsim.model.lm_head.param_quantizers["weight"] = None

# Increase bitwidth for rmsnorm due to the op having higher quantization sensitivity

# use later to change senstivity
for name, qmodule in quantsim.named_qmodules():
    if re.search(r'rmsnorm', qmodule.__class__.__name__.lower()):
        qmodule.param_quantizers['weight'] = QuantizeDequantize(shape=(), bitwidth=16, symmetric=False).to(next(quantsim.model.parameters()).device)


print("=" * 80)
print("PART 3.5: Adascale")
print("=" * 80)

from aimet_torch.experimental.adascale import apply_adascale

with place_model(quantsim.model, "cuda"):
    apply_adascale(qsim=quantsim,
                   data_loader=adascale_train_dataloader,
                   forward_fn=custom_forward,
                   num_iterations=adascale_iterations)


print("=" * 80)
print("PART 4: Eval and export model")
print("=" * 80)



print("=" * 80)
print("PART 4.1: Adascale eval")
print("=" * 80)

from aimet_torch.v2.utils import remove_activation_quantizers

if run_ppl_eval:
    # with event_marker("AdaScale FP model eval"):
    with place_model(quantsim.model, torch.device('cuda')), remove_activation_quantizers(quantsim.model):
        adascaled_ppl = llm_evaluate_ppl_with_dataloader(model=quantsim.model, dataloader=wikitext_test_dataloader, num_batches=num_eval_batches)
    print(f"PPL score of AdaScale model = {adascaled_ppl}")


print("=" * 80)
print("PART 4.2: Export Model")
print("=" * 80)


fp_ada_model = QuantizationSimModel.get_original_model(quantsim.model, qdq_weights = True)
fp_ada_model.save_pretrained(adascale_dir)
tokenizer.save_pretrained(adascale_dir)

# Summary

from genai_lib.common.debug.profiler import EventProfiler
EventProfiler().report()
EventProfiler().json_dump(os.path.join(adascale_dir, 'profiling_stats.json'))



'''
Qwen 3 script
'''


print("=" * 80)
print("PART 1.1: Notebook configs")
print("=" * 80)

print("=" * 80)
print("PART 1.1.1: Feature knobs")
print("=" * 80)

import os
context_length = get_config_value("QWEN3_CONTEXT_LENGTH", 3073, "int")
enable_right_padding = get_config_value("QWEN3_ENABLE_RIGHT_PADDING", False, "bool")
enable_masked_softmax = get_config_value("QWEN3_ENABLE_MASKED_SOFTMAX", True, "bool")
enable_fptquant = get_config_value("QWEN3_ENABLE_FPTQUANT", False, "bool")
enable_spinquant = get_config_value("QWEN3_ENABLE_SPINQUANT", False, "bool")
enable_lora = get_config_value("QWEN3_ENABLE_LORA", False, "bool")
freeze_base_encodings = get_config_value("QWEN3_FREEZE_BASE_ENCODINGS", True, "bool")
freeze_lora_encodings = get_config_value("QWEN3_FREEZE_LORA_ENCODINGS", False, "bool")
enable_eaglet = get_config_value("QWEN3_ENABLE_EAGLET", False, "bool")
pad_to_left = not enable_right_padding


print("=" * 80)
print("PART 1.1.2: Notebook configs")
print("=" * 80)

apply_decoder_seqmse = get_config_value("QWEN3_APPLY_DECODER_SEQMSE", False, "bool")
apply_lm_head_seqmse = get_config_value("QWEN3_APPLY_LM_HEAD_SEQMSE", False, "bool")
apply_decoder_lpbq = get_config_value("QWEN3_APPLY_DECODER_LPBQ", False, "bool")
apply_lm_head_lpbq = get_config_value("QWEN3_APPLY_LM_HEAD_LPBQ", False, "bool")
embedding_table_bitwidth = get_config_value("QWEN3_EMBEDDING_TABLE_BITWIDTH", 16, "int")
activation_bitwidth = get_config_value("QWEN3_ACTIVATION_BITWIDTH", 16, "int")
weight_bitwidth = get_config_value("QWEN3_WEIGHT_BITWIDTH", 4, "int")
num_calibration_batches = get_config_value("QWEN3_NUM_CALIBRATION_BATCHES", 200, "int")
num_seqmse_batches = get_config_value("QWEN3_NUM_SEQMSE_BATCHES", 20, "int")
num_seqmse_candidates = get_config_value("QWEN3_NUM_SEQMSE_CANDIDATES", 20, "int")
num_eval_batches = get_config_value("QWEN3_NUM_EVAL_BATCHES", 0, "int")
enable_2_4bit_mixed_precision = get_config_value("QWEN3_ENABLE_2_4BIT_MIXED_PRECISION", False, "bool")
zero_point_shift = float(get_config_value("QWEN3_ZERO_POINT_SHIFT", 0, "int"))

print(enable_masked_softmax)
print(zero_point_shift)



print("=" * 80)
print("PART 1.1.3: Speed Knobs")
print("=" * 80)

enable_fp16 = get_config_value("QWEN3_ENABLE_FP16", False, "bool")
run_ppl_eval = get_config_value("QWEN3_RUN_PPL_EVAL", True, "bool")

assert context_length <= 4096, "Context length longer than 4096 for Qwen3 model family has not been validated for accuracy"
assert not (apply_decoder_lpbq and apply_lm_head_lpbq), "Applying LPBQ to both Decoder and LM-Head has not been validated for accuracy"
assert embedding_table_bitwidth in (8, 16), "Only 8-bit and 16-bit Embedding Table have been validated"
assert not enable_fp16, "FP16 based quantization has not been tested"
assert freeze_base_encodings if freeze_lora_encodings else True, "When LoRA encodings are frozen for all concurrencies then base encodings must be frozen"
assert not enable_fptquant, "Hadmard feature not enabled in this NB yet!"
assert not (weight_bitwidth==2 and enable_2_4bit_mixed_precision), "Enable 2/4 bit mixed precision only when general weight bitwidth is set to 4 bit!"
assert not (enable_fptquant and enable_spinquant), "FPTQuant and SpinQuant are mutually exclusive, cannot enable both simultaneously"



print("=" * 80)
print("PART 1.3: Setting NSP Target")
print("=" * 80)



from utilities.nsptargets import NspTargets

# setup Target platform and its generation
TARGET_PLATFORM = get_config_value("QWEN3_TARGET_PLATFORM", "Windows").capitalize()

# Android GEN4 and GEN5 is supported for this notebook
PLATFORM_GEN = get_config_value("PLATFORM_GEN", 3, "int")
nsp_target = eval(f"NspTargets.{TARGET_PLATFORM}.GEN{PLATFORM_GEN}")

# Select quantsim config based on target
htp_config_file = f'htp_quantsim_config_{nsp_target.dsp_arch}_per_channel_linear.json' #This was initially



print("=" * 80)
print("PART 2: Instantiate and Eval HF FP Model")
print("=" * 80)



import torch
torch.set_grad_enabled(False)
from transformers.models.qwen3 import modeling_qwen3
from aimet_torch.utils import place_model
from genai_lib.common.debug.profiler import event_marker
from genai_lib.common.debug.recipe_logger import llm_lib_log_env_info, recipe_dump_init

model_name = get_config_value("QWEN3_MODEL_NAME", 'qwen3')
model_id = get_config_value("QWEN3_MODEL_ID", adascale_dir)
if enable_eaglet:
    draft_id = get_config_value("QWEN3_DRAFT_ID", None, "none")
cache_dir = get_config_value("CACHE_DIR", './cache_dir')
output_dir = get_config_value("OUTPUT_DIR", "./output_dir")
os.makedirs(output_dir, exist_ok=True)
recipe_dump_init(output_dir, "genai_lib_debug")
llm_lib_log_env_info()



print("=" * 80)
print("PART 2.1: Load HF Model")
print("=" * 80)


from transformers import AutoConfig, AutoTokenizer
llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
num_hidden_layers = get_config_value("QWEN3_NUM_HIDDEN_LAYERS", 0, "int")
llm_config.num_hidden_layers = num_hidden_layers if num_hidden_layers > 0 else llm_config.num_hidden_layers
print(f'num_layer: {llm_config.num_hidden_layers}, context_length: {context_length}, '
      f'num_hidden_size: {llm_config.num_attention_heads}, num_kv_heads: {llm_config.num_key_value_heads}')

if enable_eaglet:
    draft_config = AutoConfig.from_pretrained(draft_id, cache_dir=cache_dir, trust_remote_code=True)
model = modeling_qwen3.Qwen3ForCausalLM.from_pretrained(model_id, config=llm_config)

# model.config.return_dict = True

os.environ['TOKENIZERS_PARALLELISM'] = '0'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
# Adjust the tokenizer to limit to context_length
tokenizer.model_max_length = context_length

# Reduce the precision of the model to FP16 to minimize the amount of GPU memory needed
if enable_fp16:
    model.half()


print("=" * 80)
print("PART 2.2: Instantiate Dataloaders")
print("=" * 80)


from llm_utils.wikitext_dataloader import get_wiki_dataset

valid_datasets = {}


with event_marker("Instantiate wikitext Dataloaders"):
    wiki_train_dataloader, wiki_test_dataloader, wiki_dataset = get_wiki_dataset(context_length, tokenizer, cache_dir, path=get_config_value('QWEN3_WIKI_DATASET_PATH', None, "none"))



valid_datasets["WIKITEXT"] = {
    "dataloader": wiki_train_dataloader,
    "dataset": wiki_dataset
}





if enable_lora:
    from llm_utils.xlam_dataloader import xLAMDataset
    # with event_marker("Instantiate xLAM Dataloders"):
    xlam_train_dataloader, xlam_test_dataloader, xlam_dataset = xLAMDataset(tokenizer=tokenizer,
                                                                                block_size=context_length,
                                                                                batch_size=1).get_xlam_dataloader(path=get_config_value('QWEN3_XLAM_DATASET_PATH', ''))
    valid_datasets["XLAM"] = {
        "dataloader": xlam_train_dataloader,
        "dataset": xlam_dataset
    }

base_calibration_key = get_config_value("QWEN3_BASE_CALIBRATION_DATASET", "WIKITEXT").upper()

assert base_calibration_key in valid_datasets, (
   f"`BASE_CALIBRATION_DATASET` must be one of {list(valid_datasets)}, "
   f"but got {base_calibration_key}"
)

base_calibration_dataloader = valid_datasets[base_calibration_key]["dataloader"]
print("Using base calibration dataset:", base_calibration_key)


print("=" * 80)
print("PART 2.3: Eval HF Model")
print("=" * 80)


from genai_lib.llm.evaluation_utils import llm_evaluate_ppl_with_dataloader
from genai_lib.common.debug.recipe_logger import llm_lib_log_property, Property
from genai_lib.common.debug.recipe_logger import llm_lib_log_metric, ModelType, Metric


if run_ppl_eval:
    with place_model(model, torch.device('cuda')):
        orig_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wiki_test_dataloader, num_batches=num_eval_batches)
    print(f"PPL score of HuggingFace FP model = {orig_ppl}")
    llm_lib_log_metric(ModelType.hf_model, Metric.ppl, orig_ppl)


# Remove the HuggingFace model from memory
del model

print("=" * 80)
print("PART 3: Instantiate and Adapt FP32 Model - Monkey patching")
print("=" * 80)


print("=" * 80)
print("PART 3.1: Adapt FP32 Model for Inference on HTP")
print("=" * 80)

'''
### 3. Instantiate and Adapt FP32 model

#### 3.1 Adapt FP32 Model Definition for Inference on HTP.
- The following adaptations are done to replace default attention module with attention definition that compatible with NSP backend
  * use conv instead of linear for Q,K,V,O projections
  * bypass attention and causal mask generation and replace with pre-generated 2D-mask input
  * output only newly created V and transposed K instead of entire augmented KV sequence
  * input pre-calculated positional embedding instead of position ids, thus bypass the embedding generation in the model
'''


from transformers import cache_utils

from genai_lib.llm.dev.model_adaptation.qwen3.adaptation import (
    QcQwen3Attention,
    QcQwen3ForCausalLM,
    adapted_RotaryEmbedding,
    DynamicCache_update,
    DynamicCache_get_seq_length,
    update_attr,
    DynamicCache_to_legacy_cache,

)

with event_marker("Apply adaptations to model definition"):
    modeling_qwen3.Qwen3Attention = QcQwen3Attention
    modeling_qwen3.Qwen3ForCausalLM = QcQwen3ForCausalLM

    # Bypass rotary_emb module
    assert hasattr(modeling_qwen3.Qwen3RotaryEmbedding, 'forward'), \
    f"Unknown Qwen3RotaryEmbedding definition: {modeling_qwen3.Qwen3RotaryEmbedding}"
    modeling_qwen3.Qwen3RotaryEmbedding.forward = adapted_RotaryEmbedding

    # Adapting KV$ management
    assert update_attr(cache_utils.DynamicCache, 'update', DynamicCache_update), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
    assert update_attr(cache_utils.DynamicCache, 'get_seq_length', DynamicCache_get_seq_length),  f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
    assert update_attr(cache_utils.DynamicCache, 'to_legacy_cache', DynamicCache_to_legacy_cache), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"


print("=" * 80)
print("PART 3.2: Instantiate Adapted FP32 model definition")
print("=" * 80)

ARN = get_config_value("QWEN3_ARN", 2073, "int")
llm_lib_log_property({Property.ARN : ARN})
MASK_NEG = get_config_value("QWEN3_MASK_NEG", -200, "int")
setattr(llm_config, 'return_new_key_value_only', True)
setattr(llm_config, 'transposed_key_cache', True)
setattr(llm_config, 'use_position_embedding_input', True)
setattr(llm_config, '_attn_implementation', 'eager')
setattr(llm_config, '_attn_implementation_internal', 'eager')
setattr(llm_config, 'return_dict', False)
setattr(llm_config, 'logits_to_keep', 0)
setattr(llm_config, 'input_tokens_per_inference', ARN)
setattr(llm_config, 'enable_masked_softmax', enable_masked_softmax)
setattr(llm_config, 'use_cache', True)
llm_config.save_pretrained(output_dir)
# Recipe_logger: Log the ARN of the prepared model
llm_lib_log_property({Property.ARN : ARN})


with event_marker('Adapted FP model creation'):
    model = modeling_qwen3.Qwen3ForCausalLM.from_pretrained(model_id, config=llm_config)


if enable_spinquant:
    print("=" * 80)
    print("PART 3.2: Apply Rotations - Spinquant")
    print("=" * 80)

with event_marker("Apply Rotations"):
    if enable_spinquant:
        from aimet_torch.experimental.spinquant.spinquant_optimizer import SpinQuant

        # Manually untie embedding weights after init, as 'tie_word_embeddings=False' on config during init results in LM-Head initialized with random weights
        new_lm_head = torch.nn.Linear(llm_config.hidden_size, llm_config.vocab_size, bias=False)
        new_lm_head.weight.data = model.get_output_embeddings().weight.data.clone()
        model.lm_head = new_lm_head
        model.tie_word_embeddings = False

        spinquant = SpinQuant()
        spinquant.apply_spinquant(model)

    elif enable_fptquant:
    # In AIMET's FPTQuant the model being loaded has had the second half of a Hadamard merged onto the down-proj
    # Hence we have to add the first half as an online rotation in the graph, so that they cancel each other
        from aimet_torch.experimental.fptquant.fptquant_transforms import GroupedHadamardTransformOp

        def largest_power_of_two_dividing(n):
            if n <= 0:
                raise ValueError("Input must be a positive integer.")

            power = 1
            while n % (power * 2) == 0:
                power *= 2
            return power

        for layer in model.model.language_model.layers:
            intermediate_size = largest_power_of_two_dividing(layer.mlp.down_proj.weight.shape[1])
            hadamard = GroupedHadamardTransformOp(intermediate_size=intermediate_size)
            layer.mlp.down_proj = torch.nn.Sequential(hadamard, layer.mlp.down_proj)



print("=" * 80)
print("PART 3.3: Completing last steps of Model Adaptions")
print("=" * 80)

print("Converting Linear Layers to conv layers")

from genai_lib.common.dev.model_adaptation.linear_to_conv import replace_linears_with_convs

with event_marker('Model adaptation for NSP backend'):
    model = replace_linears_with_convs(model)



print("=" * 80)
print("PART 3.4: Changes to HF Model to make it work with the adapted model")
print("=" * 80)

'''
#### 3.4 Changes to HuggingFace Model to Work with Adapted Model or Prepared Model

As a result of adapting the model we introduce changes to the types of the model inputs.
As a result of model preparation, we make the shapes of the inputs static.

The forward function supplied by the original HuggingFace model can no longer be used when we are working with the prepared model.

Override the 'forward' function to make the prepared model work just like the old model.

'''




from genai_lib.llm.static_graph_utils import llm_pad_inputs, llm_create_1d_attn_mask, llm_pad_past_kv, \
    llm_trim_pad_logits, llm_get_position_ids_from_attention_mask, llm_pad_input_attn_mask, llm_create_kv_attn_mask, llm_get_dummy_kv, \
    llm_pad_position_ids
from genai_lib.llm.dev.model_adaptation.qwen3.utils import llm_create_causal_mask, llm_create_position_embeddings
from genai_lib.llm.dev.model_adaptation.common.utils import KEY_CONCAT_AXIS, VALUE_CONCAT_AXIS, llm_update_kv_cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import types
import inspect
def get_kv_length(past_key_values):
    if past_key_values is None:
        kv_length = 0
    elif not isinstance(past_key_values, tuple):
        kv_length = past_key_values.get_seq_length()
    else:
        kv_length = past_key_values[0][1].shape[-2]
    return kv_length


# Make
def prepare_inputs_for_dynamic_shapes(_model, input_ids_slice, attn_mask_slice, position_ids_slice, outputs, **kwargs):
    device = input_ids_slice.device
    batch_size = input_ids_slice.shape[0]

    kv_length = get_kv_length(outputs['past_key_values'])

    past_kv_attn_mask = torch.ones((batch_size, kv_length), dtype=torch.long, device=device)

    prepared_1d_attention_mask = llm_create_1d_attn_mask(attn_mask_past_kv = past_kv_attn_mask,
                                                         attn_mask_input = attn_mask_slice)


    prepared_causal_mask = llm_create_causal_mask(prepared_1d_attn_mask = prepared_1d_attention_mask,
                                                  input_tensor = input_ids_slice,
                                                  max_input_tokens = input_ids_slice.shape[-1],
                                                  model_context_len = context_length,
                                                  config = _model.config,
                                                  mask_neg = MASK_NEG,
                                                  cache_index = None,)

    prepared_position_embeddings = llm_create_position_embeddings(config = _model.config,
                                                                  position_ids = position_ids_slice)

    prepared_inputs = {
        'input_ids': input_ids_slice,
        'attention_mask': prepared_causal_mask,
        'position_ids': prepared_position_embeddings,
        'past_key_values': outputs['past_key_values'],
    }

    return prepared_inputs



# Make HF Model okay with Static shapes

def prepare_inputs_for_static_shapes(_model, input_ids_slice, attn_mask_slice, position_ids_slice, outputs):
    batch_size = input_ids_slice.shape[0]
    pad_token = tokenizer.eos_token_id
    device = input_ids_slice.device
    head_dim = _model.config.head_dim if hasattr(_model.config, 'head_dim') else _model.config.hidden_size // _model.config.num_attention_heads

    ####### input id preparation #######
    pad_input_ids = llm_pad_inputs(pad_token = pad_token,
                                   max_input_tokens = ARN,
                                   input_ids_slice = input_ids_slice,
                                   pad_to_left = pad_to_left)

    ####### KV input preparation #######
    dummy_kv = llm_get_dummy_kv(batch_size = batch_size,
                                num_key_value_heads = _model.config.num_key_value_heads,
                                head_dim = head_dim,
                                key_concat_axis = KEY_CONCAT_AXIS,
                                device = device,
                                cache_len = context_length-ARN if pad_to_left else context_length)

    padded_past_kv_in = llm_pad_past_kv(dummy_past_kv = dummy_kv,
                                        unpadded_past_kv = outputs['past_key_values'],
                                        num_hidden_layers = _model.config.num_hidden_layers,
                                        key_concat_axis = KEY_CONCAT_AXIS,
                                        value_concat_axis = VALUE_CONCAT_AXIS,
                                        pad_to_left = pad_to_left,
                                       )

    ######### Attention mask Input preparation #######
    inp_attn_mask = llm_pad_input_attn_mask(attn_mask_slice = attn_mask_slice,
                                            max_input_tokens = ARN,
                                            pad_to_left = pad_to_left)

    kv_length = get_kv_length(outputs['past_key_values'])

    past_kv_attn_mask = llm_create_kv_attn_mask(unpadded_past_kv = outputs['past_key_values'],
                                                model_context_len = context_length,
                                                max_input_tokens = ARN,
                                                batch_size = batch_size,
                                                device = device,
                                                pad_to_left = pad_to_left)

    if pad_to_left:
        cache_index = None
    else:
        cache_index = torch.tensor([kv_length], dtype=torch.int64, device=device)

    prepared_1d_attention_mask = llm_create_1d_attn_mask(attn_mask_past_kv = past_kv_attn_mask,
                                                         attn_mask_input = inp_attn_mask,
                                                         cache_index = cache_index)

    # due to model adaptation
    prepared_causal_mask = llm_create_causal_mask(prepared_1d_attn_mask = prepared_1d_attention_mask,
                                                  input_tensor = pad_input_ids,
                                                  max_input_tokens = ARN,
                                                  model_context_len = context_length,
                                                  config = _model.config,
                                                  mask_neg = MASK_NEG,
                                                  cache_index = cache_index,
                                                  pad_to_left = pad_to_left)
    ########### Position ID preparation #######
    padded_position_ids = llm_pad_position_ids(position_ids_slice = position_ids_slice,
                                                max_input_tokens = ARN,
                                                pad_to_left = pad_to_left)
    # model adaptation
    prepared_position_embeddings = llm_create_position_embeddings(config = _model.config,
                                                                  position_ids = padded_position_ids)

    prepared_inputs = {
        'input_ids': pad_input_ids,
        'attention_mask': prepared_causal_mask,
        'position_ids': prepared_position_embeddings,
        'past_key_values': padded_past_kv_in
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
    position_ids=None,
    past_key_values=None,
    return_dict=False,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=False,
    cache_position=None,
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
    outputs = {'past_key_values': past_key_values}
    for i, input_slice in enumerate(slice_inputs_gen_obj):
        if num_slices is not None and i >= num_slices:
            break
        input_ids_slice = input_slice['input_ids']
        attn_mask_slice = input_slice['attention_mask']
        position_ids_slice = input_slice['position_ids']
        if static_shape:
            prepared_inputs = prepare_inputs_for_static_shapes(self,
                                                               input_ids_slice = input_ids_slice,
                                                               attn_mask_slice = attn_mask_slice,
                                                               position_ids_slice = position_ids_slice,
                                                               outputs = outputs)

        else:
            prepared_inputs = prepare_inputs_for_dynamic_shapes(self,
                                                                input_ids_slice = input_ids_slice,
                                                                attn_mask_slice = attn_mask_slice,
                                                                position_ids_slice = position_ids_slice,
                                                                outputs = outputs)

        cur_outputs = self.model(**prepared_inputs)
        if not static_shape:
            cur_outputs = (self.lm_head(cur_outputs[0]),) + cur_outputs[1:]

        outputs['past_key_values'] = llm_update_kv_cache(unpadded_past_kv = outputs['past_key_values'],
                                                         current_key_values = cur_outputs[1],
                                                         key_concat_axis = KEY_CONCAT_AXIS,
                                                         value_concat_axis = VALUE_CONCAT_AXIS,
                                                         input_ids_slice = input_ids_slice,
                                                         pad_to_left = pad_to_left)

        lm_logits = llm_trim_pad_logits(cur_logits = cur_outputs[0],
                                        input_ids_slice = input_ids_slice,
                                        pad_to_left = pad_to_left)

        bsz, _, dim = lm_logits.shape

        outputs['logits'] = torch.cat(
                (outputs.get('logits', torch.zeros((bsz, 0, dim), device=lm_logits.device)), lm_logits),
                dim=1)

        if output_hidden_states:
            last_hidden_states = llm_trim_pad_logits(cur_logits = cur_outputs[2][-1],
                                                     input_ids_slice=input_ids_slice,
                                                     pad_to_left = pad_to_left)
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
            attentions=None,
        )
    return tuple(outputs.get(out) for out in ['loss', 'logits', 'past_key_values', 'hidden_states', 'attentions'] if outputs.get(out) is not None)




print("=" * 80)
print("PART 3.5: Eval Adapted Model")
print("=" * 80)



if run_ppl_eval:
    model.forward = types.MethodType(adapted_model_forward, model)
    with event_marker("Evaluate base adapted model"):
        with place_model(model, torch.device('cuda')):
            adapted_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wiki_test_dataloader, num_batches=num_eval_batches)
    print(f"PPL score of adapted model = {adapted_ppl}")
    model.forward = types.MethodType(QcQwen3ForCausalLM.forward, model)
    llm_lib_log_metric(ModelType.adapted_model, Metric.ppl, adapted_ppl)



print("=" * 80)
print("PART 4: Model Sample Input")
print("=" * 80)

def get_dummy_data(device="cuda"):
    input_ids = torch.randint(0, len(tokenizer), (1, ARN), device=device)
    attn_mask = torch.ones((1, ARN), device=device, dtype=torch.long)
    position_ids = torch.randint(0, len(tokenizer), (1, ARN), device=device)
    outputs = {"past_key_values": None}
    with place_model(model, device):
        dummy_input = prepare_inputs_for_static_shapes(model, input_ids, attn_mask, position_ids, outputs)
    return dummy_input





print("=" * 80)
print("PART 5: Prepare Model for QAIRT Model Preparer Pro")
print("=" * 80)


print("=" * 80)
print("PART 5.1: KV Cache MHA Model Preparation")
print("=" * 80)

if enable_eaglet:
    from genai_lib.common.dev.utils import change_signature_defaults
    model.forward = change_signature_defaults(func=model.forward, defaults_dict={"output_hidden_states": True})
    output_index_filter = [":", ":", -1] # all logits, all kv-cache, last hidden states
    setattr(model.config, "output_index_filter", output_index_filter)


from qti.aisw.emitter.utils.torch_utils import load_torch_model_using_safetensors
from genai_lib.llm.model_preparation_utils import llm_build_preparer_converter_args
from genai_lib.llm.utils import llm_model_input_output_names
from qti.aisw.preparer_api.model_preparer import prepare_model

# Configuring the model for KVCache mode
model.num_logits_to_return = ARN

prepare_path = os.path.join(output_dir, 'prepare')
os.makedirs(prepare_path, exist_ok=True)
prepare_filename = f'{model_name}_kvcache_{llm_config.num_hidden_layers}_layer'

skip_prepare = get_config_value("QWEN3_SKIP_PREPARE", False, "bool")
if not skip_prepare:
    dummy_input = get_dummy_data(device=model.model.device)
    input_names, output_names = llm_model_input_output_names(llm_config.num_hidden_layers)
    converter_args = llm_build_preparer_converter_args(llm_config.num_hidden_layers, input_names, use_qairt_mpp=True) # Build converter args

    if enable_right_padding:
        input_names += ["cache_index"]
    if enable_eaglet:
        output_names += ['last_hidden_states']
        # converter_args['enable_framework_trace'] = True # needed for generating HF QDQ model for Draft training

    with event_marker("KVCache Prepare Model", flush_ram=True):

        if __name__ == '__main__': # We use the main guard to prevent child processes from re-running the top-level code
            _ = prepare_model(model,
                              dummy_input,
                              model_name = prepare_filename,
                              filename = prepare_filename,
                              path = prepare_path,
                              input_names = input_names,
                              output_names = output_names,
                              onnx_export_args = {"opset_version":20},
                              converter_args = converter_args,
                              keep_original_model_structure = False, # Flatten the model to enable weight-sharing by setting
                              order_inputs = True,
                              order_outputs = True,
                              skipped_optimizers = ['eliminate_common_subexpression',
                                                   'eliminate_nop_with_unit',
                                                   'eliminate_duplicate_initializer'
                                                   ],
                               return_prepare_model = enable_lora
                               )
            if enable_lora:
                del _




print("=" * 80)
print("PART 5.2: Delete Adapted model and lmhead")
print("=" * 80)

del model.model
del model.lm_head

model.model = None
model.lm_head = None



print("=" * 80)
print("PART 6: Eval of Prepared model")
print("=" * 80)

print("=" * 80)
print("PART 6.1: Changes to hf model to work with the adapted model")
print("=" * 80)


import time

with event_marker(f"Load pre-prepared {prepare_filename}", flush_ram=True):
    prepared_model_path = os.path.join(prepare_path, f'{prepare_filename}.py')
    if not os.path.exists(prepared_model_path):
        raise ValueError(f"prepared artifacts not found in {prepare_path}")
    elif skip_prepare:
        print(f'Preparation skipped for model={prepare_filename}, prepared at {time.ctime(os.path.getmtime(prepared_model_path))}')
    prepared_model = load_torch_model_using_safetensors(path=prepare_path, filename=prepare_filename, model_name=prepare_filename)

model.model = prepared_model
if enable_eaglet:
    del model.config.output_index_filter
model.forward = types.MethodType(adapted_model_forward, model)


print("=" * 80)
print("PART 6.2: Convert the model to half precision ")
print("=" * 80)

if enable_fp16:
    torch.set_default_dtype(torch.float16)
    model.half()


print("=" * 80)
print("PART 6.3: Eval of PPL score on the prepared model ")
print("=" * 80)

if run_ppl_eval:
    with event_marker("Evaluate base prepared model", flush_ram=True):
        with place_model(model, torch.device("cuda")):
            prepared_kvcache_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wiki_test_dataloader, num_batches=num_eval_batches)

    # This should be very close (<1e-4 delta) to original model's perplexity
    # If the perplexity score goes further up, it indicates the AIMET/QNN pair is producing a faulty prepared model
    print(f"ppl score of KVCACHE prepared fp model = {prepared_kvcache_ppl}")
    print(f"Diff between HF adapted ppl and prepared ppl = {adapted_ppl - prepared_kvcache_ppl}")
    llm_lib_log_metric(ModelType.prepared_model, Metric.ppl, prepared_kvcache_ppl)



print("=" * 80)
print("PART 7: Quantization")
print("=" * 80)


print("=" * 80)
print("PART 7.1: Create Quantsim configured for QNN HTP Target ")
print("=" * 80)


from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel

if apply_lm_head_seqmse or apply_decoder_seqmse or enable_lora:
    from utilities.model_utils import copy_model_with_shared_weights
    fp_prepared_model = copy_model_with_shared_weights(prepared_model)

# weight_bitwidth = 8
dummy_input = get_dummy_data(device = "cuda")
sig = inspect.signature(prepared_model.forward)
dummy_input_sorted = {}
for key in list(sig.parameters.keys()):
    dummy_input_sorted[key] = dummy_input[key]
dummy_input = tuple(dummy_input_sorted.values())
# weight_bitwidth= 8
with event_marker("Create Quantsim"):
    with place_model(prepared_model, "cuda"):
        quantsim = QuantizationSimModel(model=prepared_model,
                                        quant_scheme=QuantScheme.post_training_tf,
                                        dummy_input=dummy_input,
                                        default_output_bw=activation_bitwidth,
                                        default_param_bw=weight_bitwidth,
                                        in_place=True,
                                        config_file=htp_config_file)



print("=" * 80)
print("PART 7.2: Setting 16bit x 8bit matmuls")
print("=" * 80)

from aimet_torch.v2.experimental.quantsim_utils import set_matmul_second_input_producer_to_8bit_symmetric
set_matmul_second_input_producer_to_8bit_symmetric(quantsim)



print("=" * 80)
print("PART 7.3: Concat encoding unification")
print("=" * 80)

from aimet_torch.v2.experimental import propagate_output_encodings
from aimet_torch.nn.modules import custom as aimet_ops

propagate_output_encodings(quantsim, aimet_ops.Concat)





print("=" * 80)
print("PART 7.4: Manual Mixed Precision")
print("=" * 80)


import re
import json
from llm_utils.mixed_precision_overrides import ManualQuantsimMixedPrecisionConfig
from aimet_torch.v2.nn.modules.custom import QuantizedRmsNorm
from aimet_torch.v2.quantization.affine import QuantizeDequantize,AffineQuantizerBase
from aimet_torch.v2.nn.true_quant import QuantizedConv2d

def apply_manual_mixed_precision(sim):
    for name,module in sim.model.named_modules():
        if isinstance(module, AffineQuantizerBase) and module.bitwidth == 2:
            module.zero_point_shift = zero_point_shift
    if weight_bitwidth==4 and enable_2_4bit_mixed_precision:
        two_bit_patterns = [".*up_proj", ".*gate_proj"]
        # setting selected conv to 2bit , the rest 4bit
        for name, module in sim.model.named_modules():
            if isinstance(module, QuantizedConv2d) and any(re.match(pattern, name) for pattern in two_bit_patterns):
                module.param_quantizers.weight = QuantizeDequantize(shape=module.param_quantizers.weight.shape, bitwidth=4, symmetric=True,zero_point_shift=zero_point_shift)

     # if weight_bitwidth==4 and enable_2_4bit_mixed_precision:
     #    two_bit_patterns = [".*up_proj", ".*gate_proj"]
     #    # setting selected conv to 2bit , the rest 4bit
     #    for name, module in sim.model.named_modules():
     #        if isinstance(module, QuantizedConv2d) and any(re.match(pattern, name) for pattern in two_bit_patterns):
     #            module.param_quantizers.weight = QuantizeDequantize(shape=module.param_quantizers.weight.shape, bitwidth=2, symmetric=True,zero_point_shift=zero_point_shift)

    with open("./mixed_precision_config/exceptions.json", "r") as f_in:
        mixed_precision_config = json.load(f_in)

    print(mixed_precision_config)

    # Customize mixed precision config based on user parameters
    for entry in mixed_precision_config['name_list']:
        if "model_embed_tokens_Gather" in entry['module_name']:
            entry['exceptions']['param_exceptions']['bitwidth'] = embedding_table_bitwidth

    quantsim_adjuster = ManualQuantsimMixedPrecisionConfig(mixed_precision_config_file = mixed_precision_config)
    quantsim_adjuster.apply_exceptions(sim)

    # Make RMSNorm encodings per-tensor (they default to per-channel)
    for name, qmodule in sim.named_qmodules():
        if isinstance(qmodule, QuantizedRmsNorm):
            qmodule.param_quantizers['weight'] = QuantizeDequantize(shape=(), bitwidth=16, symmetric=False)

    if enable_eaglet and not getattr(draft_config, 'dual_fc', False):
        # Embedding must match encodings from last_hidden_states (which are the last RMSNorm's output encodings)
        for name, module in sim.model.named_modules():
            if "rms_norm" in name and "." not in name:
                last_rms_norm = module
        sim.model.model_embed_tokens_Gather.param_quantizers['weight'] = last_rms_norm.output_quantizers[0]

    if enable_masked_softmax:
        # Simulate SuperGroup for the following ops on the masked_softmax pattern:
        # (QK-MatMul →) FakeQuantizedAMin → Add ↘
        #                                       FakeQuantizedWhere → Softmax
        # Attention Mask → FakeQuantizedEqual ↗
        from aimet_torch.v2.nn.modules import custom as aimet_ops
        for name, qmodule in sim.named_qmodules():
            if "self_attn_Add_2" in name or isinstance(qmodule, (aimet_ops.AMin, aimet_ops.Where)):
                for idx in range(len(qmodule.input_quantizers)):
                    qmodule.input_quantizers[idx] = None
                for idx in range(len(qmodule.output_quantizers)):
                    qmodule.output_quantizers[idx] = None

    # Disable encodings on Hadamard module, as Hadamard has it's own backend FHT implementation which does not
    # use the weights to generate the hamamard values, hence on-target values are equivalent to floating-point ones.
    if enable_fptquant:
        for name, qmodule in sim.named_qmodules():
            # Hadamard op
            if "mlp_down_proj_down_proj_0_hadamard_Conv" in name:
                print(f"Disabling {name} quantizers")
                qmodule.output_quantizers[0] = None
                qmodule.param_quantizers["weight"] = None
            # Optional seed applied to Hadamard
            if name.endswith("mlp_down_proj_down_proj_0_Mul"):
                print(f"Disabling {name} input quantizers")
                for idx in range(len(qmodule.input_quantizers)):
                    qmodule.input_quantizers[idx] = None

apply_manual_mixed_precision(quantsim)




print("=" * 80)
print("PART 7.5: Apply Block Quantization")
print("=" * 80)


from aimet_torch.v2.nn.true_quant import QuantizedConv2d
from aimet_torch.v2.quantsim.config_utils import set_grouped_blockwise_quantization_for_weights

def apply_lpbq(sim):
    lpbq_conditions = []
    hadamard_substrings = ("R3", "R4", "hadamard")
    hadamard_modules = [module for name, module in sim.model.named_modules() if any(substring in name for substring in hadamard_substrings)]

    if apply_decoder_lpbq:
        lpbq_conditions.append(lambda module: module not in hadamard_modules and isinstance(module, QuantizedConv2d) and module.param_quantizers['weight'].bitwidth == 4)
    if apply_lm_head_lpbq:
        lm_head_modules = [qmodule for name, qmodule in sim.named_qmodules() if "lm_head" in name]
        lpbq_conditions.append(lambda module: module not in hadamard_modules and module in lm_head_modules and isinstance(module, QuantizedConv2d))

    arg = (lambda module: any(condition(module) for condition in lpbq_conditions)) if lpbq_conditions else None

    if arg:
        set_grouped_blockwise_quantization_for_weights(sim = sim,
                                                       arg = arg,
                                                       bitwidth = 4,
                                                       symmetric = True,
                                                       decompressed_bw = 8,
                                                       block_size = 128,
                                                       block_grouping = -1)

if apply_decoder_lpbq or apply_lm_head_lpbq:
    apply_lpbq(quantsim)
    print("apply lpbq")



print("=" * 80)
print("PART 7.6: Scatter elements OP Encoding fix")
print("=" * 80)

from aimet_torch._base.nn.modules.custom import ScatterElements, Permute

def unify_encodings(source_name, destination_name, start_layer=0, end_layer=None):
    def _find_module_dict(name):
        for module_name, module in quantsim.model.named_modules():
            if module_name.endswith(name):
                start = module_name.find(name)
                yield module_name[:start], module

    sources = { name:module for name, module in _find_module_dict(source_name) }
    destinations = { name:module for name, module in _find_module_dict(destination_name) }

    sources = dict(list(sources.items())[start_layer:end_layer])
    destinations = dict(list(destinations.items())[start_layer:end_layer])

    assert len(sources)==len(destinations) and len(sources)> 0, f"Cannot execute encoding alignment due to mismatched pairing of \
        source and destination quantizers. String matching found {len(sources)} sources, and {len(destinations)} destinations."
    # copying quantizers from source module
    for module_name, source_module in sources.items():
        destination_module = destinations[module_name]
        if isinstance(destination_module, ScatterElements):
            destination_module.input_quantizers[2] = source_module.output_quantizers[0]
            destination_module.input_quantizers[0] = source_module.output_quantizers[0]
            destination_module.output_quantizers[0] = source_module.output_quantizers[0]
        elif isinstance(destination_module, Permute):
            destination_module.output_quantizers[0] = source_module.output_quantizers[0]

if enable_right_padding:
    unify_encodings('self_attn_Concat_1', 'self_attn_ScatterElements_1', start_layer=0, end_layer=llm_config.num_hidden_layers)
    unify_encodings('self_attn_v_proj_Conv', 'self_attn_ScatterElements', start_layer=0, end_layer=llm_config.num_hidden_layers)


print("=" * 80)
print("PART 7.7: Sequential MSE")
print("=" * 80)


import math
from aimet_torch.v2.seq_mse import apply_seq_mse, SeqMseParams

def perform_seqmse(sim, fp_model):
    def _seq_mse_forward_fn(_model, inputs):
        model.model = _model
        model(**inputs)

    seqmse_dataloader_length = ARN # ensure length less than or equal to ARN to avoid not useful slicing in seqmse forward pass
    with event_marker("Instantiate wikitext Dataloaders"):
        seqmse_wiki_train_dataloader, _, _ = get_wiki_dataset(seqmse_dataloader_length, tokenizer, cache_dir, path=get_config_value('QWEN3_WIKI_DATASET_PATH', None, "none"))


    lm_head_fp_modules = [module
                          for module_name, module in fp_model.named_modules()
                          if isinstance(module, torch.nn.Conv2d) and 'lm_head' in module_name]
    decoder_fp_modules = [module
                          for module_name, module in fp_model.named_modules()
                          if isinstance(module, torch.nn.Conv2d) and 'lm_head' not in module_name]
    hadamard_modules = [module for name, module in fp_model.named_modules() if any(substring in name for substring in ("R3", "R4", "hadamard"))]

    if apply_decoder_seqmse and apply_lm_head_seqmse:
        modules_to_exclude = hadamard_modules
    elif apply_decoder_seqmse:
        modules_to_exclude = lm_head_fp_modules + hadamard_modules
    elif apply_lm_head_seqmse:
        modules_to_exclude = decoder_fp_modules + hadamard_modules

    recommended_block_size = 2048

    total_seqmse_data = recommended_block_size * num_seqmse_batches
    num_batches = math.ceil(total_seqmse_data / seqmse_dataloader_length) # recipe from system recommended optimal recipe

    seqmse_params = SeqMseParams(num_batches=num_batches,
                                 inp_symmetry='symqt',
                                 num_candidates=num_seqmse_candidates,
                                 loss_fn='mse',
                                 forward_fn = _seq_mse_forward_fn)

    with place_model(sim.model, torch.device("cuda")), place_model(fp_model, torch.device("cuda")):
        with torch.no_grad():
            apply_seq_mse(fp_model, sim, seqmse_wiki_train_dataloader, seqmse_params, modules_to_exclude=modules_to_exclude)


if apply_decoder_seqmse or apply_lm_head_seqmse:
    with event_marker("Apply SeqMSE on base model"):
        perform_seqmse(quantsim, fp_prepared_model)

    if not enable_lora:
        del fp_prepared_model




print("=" * 80)
print("PART 7.8: Calibration")
print("=" * 80)


from tqdm import tqdm
from aimet_torch.v2.experimental.quantsim_utils import clip_weights_to_7f7f

def perform_calibration(sim, calibration_dataloader, num_batches=200):
    def _calibration_forward_fn(sim_model, kwargs):
        model.model = sim_model
        data_loader = kwargs['data_loader']
        max_iterations = kwargs['num_batches']
        for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
            if batch_id < max_iterations:
                model(input_ids=batch['input_ids'].to(device=torch.device('cuda')))
            else:
                break

    kwargs = {
        'data_loader': calibration_dataloader,
        'num_batches': num_batches
    }

    with place_model(sim.model, "cuda"):
        with torch.no_grad():
            sim.compute_encodings(_calibration_forward_fn, kwargs)

    clip_weights_to_7f7f(sim)


with event_marker("Compute encoding for base model", flush_ram=True):
    perform_calibration(quantsim, base_calibration_dataloader, num_calibration_batches)


print("=" * 80)
print("PART 7.9: Eval KV Cache sim Model")
print("=" * 80)



if run_ppl_eval:
    with event_marker("Evaluate base quantsim model", flush_ram=True):
        with place_model(model, torch.device("cuda")):
            model.model = quantsim.model
            sim_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wiki_test_dataloader, num_batches=num_eval_batches)

    print(f"ppl score of KVCACHE sim fp model = {sim_ppl}")
    print(f"Diff between adapted ppl and kvcache sim ppl = {adapted_ppl - sim_ppl}")
    llm_lib_log_metric(ModelType.qsim_model, Metric.ppl, sim_ppl)



print("=" * 80)
print("PART 8: Export")
print("=" * 80)


print("=" * 80)
print("PART 8.1: Export ONNX and encodings")
print("=" * 80)

from lora_utils.lora_meta_utils import get_updatable_tensors
from aimet_torch import onnx_utils
onnx_utils.EXPORT_TO_ONNX_DIRECT = True
onnx_utils.RESTORE_ONNX_MODEL_INITIALIZERS = True

def export_onnx_and_encodings(sim, onnx_dir, filename_prefix, generate_updatable_tensors = False):
    input_names, output_names = llm_model_input_output_names(llm_config.num_hidden_layers, use_position_embedding_input=True, separate_tuple_input_output=True)

    if enable_right_padding:
        input_names += ["cache_index"]
    if enable_eaglet:
        output_names += ['last_hidden_states']
    if enable_fp16:
        # Convert FP16 model back to FP32 for ONNX export
        torch.set_default_dtype(torch.float32)
        model.float()

    dummy_input = get_dummy_data(device = "cpu")

    sig = inspect.signature(sim.model.forward)
    dummy_input_sorted = {}
    for key in list(sig.parameters.keys()):
        dummy_input_sorted[key] = dummy_input[key]
    dummy_input = dummy_input_sorted
    dummy_input = tuple(list(dummy_input.values()))

    onnx_api_args = onnx_utils.OnnxExportApiArgs(input_names=input_names, output_names=output_names, opset_version=20)

    os.makedirs(onnx_dir, exist_ok=True)
    with place_model(sim.model, torch.device("cpu")):
        sim.export(onnx_dir, filename_prefix, dummy_input, onnx_export_args=onnx_api_args,
                    filename_prefix_encodings=filename_prefix)

    if generate_updatable_tensors:
        updatable_tensors = get_updatable_tensors(sim,
                                                  os.path.join(onnx_dir, f"{filename_prefix}.onnx"),
                                                  os.path.join(onnx_dir, f"{filename_prefix}.encodings"))
        updatable_tensors_path = os.path.join(onnx_dir, f'{filename_prefix}_updatable_tensors.txt')
        with open(updatable_tensors_path, "w") as f:
            for tensor in updatable_tensors:
                f.write(str(tensor) + "\n")

with event_marker(f"Export onnx and encodings for base model", flush_ram=True):
    base_onnx_dir = os.path.join(output_dir, 'base', 'onnx')
    base_filename_prefix = f"{model_name}_base"
    export_onnx_and_encodings(quantsim, base_onnx_dir, base_filename_prefix)

tokenizer_dir = output_dir
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save_pretrained(tokenizer_dir)

# save generation config
model.generation_config.save_pretrained(output_dir)


print("=" * 80)
print("PART 8.2: Generate Test Vectors")
print("=" * 80)

from itertools import islice
from collections import deque
from genai_lib.llm.test_vectors import generate_test_vectors

def generate_test_vectors_for_usecase(sim, output_dir, num_test_vectors = 1, slice_num = 0, device = torch.device('cuda')):
    split_candidates_layers = [
        "model_embed_tokens_Gather",
        "model_layers_\\d+_Add_1$"
    ]
    idx_to_name_output_dict = {0:'logits', 1:'past_key_values'}
    if enable_eaglet:
        idx_to_name_output_dict[2] = 'last_hidden_states'
    with place_model(sim.model, device):
        for index, batch in enumerate(wiki_train_dataloader):
            if index >= num_test_vectors:
                break
            outputs = {'past_key_values': None}
            if slice_num > 0:
                model.model = sim.model
                with torch.no_grad():
                    output = model(input_ids=batch['input_ids'].to(device=device), num_slices=slice_num, return_dict=True)
                    outputs['past_key_values'] = output.past_key_values
            input_ids = batch['input_ids'].to(device)
            attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1]), dtype = torch.long, device = device)
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            # format is: "var_name": (var_tensor, slice_dim)
            tensor_dict = {'input_ids': (input_ids, 1),
                      'attention_mask': (attention_mask, 1),
                      'position_ids': (position_ids, 1)}
            slice_inputs_gen_obj = slice_tensors(slice_length = ARN,
                                                 max_length = input_ids.shape[-1],
                                                 tensor_dict = tensor_dict,
                                                 remainder_first = slice_num>0)
            slice_input = deque(islice(slice_inputs_gen_obj, slice_num + 1), maxlen=1)[0]
            model_inputs = prepare_inputs_for_static_shapes(model, slice_input['input_ids'], slice_input['attention_mask'], slice_input['position_ids'], outputs=outputs)
            generate_test_vectors(sim=sim, model_inputs=model_inputs, output_dir=output_dir,
                                  batch_index=index, test_vector_layers=split_candidates_layers, idx_to_name_output_dict=idx_to_name_output_dict)
with torch.no_grad():
    with event_marker("generate base model test vectors"):
        generate_test_vectors_for_usecase(quantsim, os.path.dirname(base_onnx_dir), slice_num=1)




print("=" * 80)
print("PART 8.3: Base model PEFT config export")
print("=" * 80)



if enable_lora:

    import onnx
    from aimet_torch.onnx_utils import OnnxSaver

    with event_marker("Generate onnx node io tensor map"):
        onnx_model = onnx.load(os.path.join(base_onnx_dir, f'{base_filename_prefix}.onnx'))
        onnx_node_to_io_tensor_map, _ = OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

        layers_to_onnx_op_names = onnx_utils.get_layers_in_io_tensor_map(onnx_node_to_io_tensor_map)

    attach_point_onnx_mapping_path = os.path.join(base_onnx_dir, f'{base_filename_prefix}_node_mapping.json')
    with open(attach_point_onnx_mapping_path, 'w') as f:
        json.dump(layers_to_onnx_op_names, f, indent=2)

    use_case_list = []

    use_case_list.append(
        {
            'name': 'base',
            'adapter_names': [],
            'model_name': f'base/onnx/{base_filename_prefix}.onnx',
            'quant_overrides': f'base/onnx/{base_filename_prefix}.encodings',
            "quant_updatable_tensors": None,
        }
    )

# SUMMARY


from genai_lib.common.debug.profiler import EventProfiler
from genai_lib.common.debug.recipe_logger import dump_logs_to_json
EventProfiler().report()
EventProfiler().json_dump(os.path.join(output_dir, 'profiling_stats.json'))
dump_logs_to_json()
