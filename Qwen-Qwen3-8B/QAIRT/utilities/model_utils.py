# ==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2025 Qualcomm Technologies, Inc. All rights reserved.
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

from copy import deepcopy
import functools

def copy_model_with_shared_weights(source_model):
    target_model = deepcopy(source_model)
    for name, source_parameter in source_model.named_parameters():
        pre, _, post = name.rpartition('.')
        pre_obj = functools.reduce(getattr, [target_model] + pre.split('.')) if pre else target_model
        setattr(pre_obj, post, source_parameter)
    return target_model
