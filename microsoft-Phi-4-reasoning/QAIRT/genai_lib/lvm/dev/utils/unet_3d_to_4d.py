#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch
import copy
from genai_lib.common.dev.utils import rsetattr
from diffusers.models.transformers.transformer_2d import Transformer2DModel as Transformer2DModel_3d
from genai_lib.lvm.dev.diffusers.transformer_2d import Transformer2DModel as Transformer2DModel_4d


def convert_3d_unet_to_4d(model: torch.nn.Module):
    model_4d = copy.deepcopy(model)

    for name, module_3d in model_4d.named_modules():
        if isinstance(module_3d, Transformer2DModel_3d):

            layers_geglu = {f'transformer_blocks.{idx}.ff.net.0.proj.weight' for idx in range(10)}
            layers_ff = {f'transformer_blocks.{idx}.ff.net.2.weight' for idx in range(10)}
            layers_proj = {'proj_in.weight', 'proj_out.weight'}
            linear_to_conv_weights = set.union(*[layers_geglu, layers_ff, layers_proj])

            state_dict_3d = module_3d.state_dict()
            layer_matches = set.intersection(linear_to_conv_weights, set(state_dict_3d.keys()))
            for layer_name in layer_matches:
                shape_2d = state_dict_3d[layer_name].shape[:2]
                state_dict_3d[layer_name] = state_dict_3d[layer_name].reshape(shape_2d)[:,:,None,None]

            module_4d = Transformer2DModel_4d(**module_3d.config)
            module_4d.load_state_dict(state_dict_3d)

            rsetattr(model_4d, name, module_4d)

    model_4d.to(model.device)
    return model_4d