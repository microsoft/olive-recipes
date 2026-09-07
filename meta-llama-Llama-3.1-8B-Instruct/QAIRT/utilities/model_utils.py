# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

from copy import deepcopy
import functools

def copy_model_with_shared_weights(source_model):
    target_model = deepcopy(source_model)
    for name, source_parameter in source_model.named_parameters():
        pre, _, post = name.rpartition('.')
        pre_obj = functools.reduce(getattr, [target_model] + pre.split('.')) if pre else target_model
        setattr(pre_obj, post, source_parameter)
    return target_model
