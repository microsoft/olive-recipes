# ==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2024 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# ==============================================================================
import functools
import warnings

import torch


def linear_to_conv2d(linear_module):
    try:
        conv2d_module = torch.nn.Conv2d(linear_module.in_features, linear_module.out_features, kernel_size=1)
        conv2d_module.weight.data.copy_(linear_module.weight[:, :, None, None])
        if linear_module.bias is not None:
            conv2d_module.bias.data.copy_(linear_module.bias)
    except AttributeError:
        warnings.warn(f'Expect input module to be an instance of torch.nn.Linear but got {type(linear_module)}, conversion won`t take any effects!')
    
    return conv2d_module


def rsetattr(obj, attr, val):
    """
    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

