#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" common definitions used by aimet extension and test utils etc """
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List
import torch


@dataclass
class ModelCfg:
    """
    Configuration for model that needs to be quantized.
    """
    model_path: str # path for model within the diffuser pipeline e.g.  'vae.decoder'
    model: torch.nn.Module # model to be quantized.
    model_input_format_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None  # input format/filtering function
    compute_sqnr_fn: Callable = None # method used for computing SQNR between FP and sim o/p.
    out_mapper_fn: Callable = None  # method used to map Tensor/Tuple(Tensor,) to FP model o/p.
    adaround_num_iterations: int = None # number of adaround interation, 0 is to reuse, None is to disable.
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    intermediate_modules_names: Optional[List[str]] = None
    checkpoints_config_file: Optional[str] = None # path for JSON file to provide checkpoints config
