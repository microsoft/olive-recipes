# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
from transformers import AutoConfig

def get_positional_encoding(
    config= None, model_id= None, cache_dir= None
):
    """
    Populate positional-embedding field in genie config (genie_config["dialog"]["engine"]["model"]["positional-encoding"])

    Input:
    config: model config from NB2 exports
    model_id: Huggingface model id in case config isn't provided
    cache_dir: Path to a directory in which a downloaded pretrained model configuration should be cached
    access_token: access token in case of gated repo (Ex: Llama-3.2-3B)

    Output:
    dictionary for populating genie_config["dialog"]["engine"]["model"]["positional-encoding"] field in genie config
    Example:
    {
        "type": "rope",
        "rope-dim": 64,
        "rope-theta": 10000,
    }
    """

    if config is None or not config.exists():
         if model_id is None:
              print("Warning: No valid model config path or model_id provided. Please provide at least one.")
              return None
         else:
              print("Warning: No valid model config path is provided. Reading from Huggingface")
              config= AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
              config = config.to_dict()
    else:
         with open(config, 'r') as f:
              config = json.load(f)

    if 'rope_theta' not in config or config['rope_theta'] is None:
        print("rope_theta not found in model config: positional_encoding type may not be rope. Only rope is currently supported")
        return None

    positional_encoding = {
        "type": "rope",
        "rope-dim": config['head_dim'] // 2,
        "rope-theta": int(config['rope_theta']),
        }

    if 'rope_scaling' in config and config['rope_scaling'] is not None:
        positional_encoding['rope-scaling'] = {k.replace('_', '-'): v for k, v in config['rope_scaling'].items()}

    return positional_encoding
