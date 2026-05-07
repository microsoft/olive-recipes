#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import re
import json
from collections import defaultdict
import os

from aimet_torch.v2.nn.base import BaseQuantizationMixin
from aimet_torch.v2.quantization.float.quantizer import FloatQuantizeDequantize
from aimet_torch.quantsim_config.builder import LazyQuantizer
from aimet_common.defs import QuantScheme, QuantizationDataType


def apply_input_output_exception(quantizer, exception):
    assert isinstance(exception, dict), f"Exception is not a dictionary type: {exception}"

    if exception.get("enabled", True) is False:
        return None

    if quantizer is None:
        quantizer = LazyQuantizer(exception.get("bitwidth", 16),
                                  'nearest',
                                  QuantScheme.post_training_tf,
                                  exception.get("asymmetric", True) is False,
                                  enabled_by_default=True,
                                  data_type=QuantizationDataType.float if exception.get("data_type", "int") == "float" else QuantizationDataType.int
                                  ).realize()

    if "bitwidth" in exception.keys():
        if isinstance(quantizer, FloatQuantizeDequantize):
            assert exception['bitwidth'] in [8, 16], "Bitwidth for FloatQuantizeDequantize should only be 8 or 16"
            # For quantizers with float dtype, we can't set value to the property "bitwidth" directly,
            # we should set the "exponent_bits" and "mantissa_bits" correspondingly
            quantizer.exponent_bits = 5 if exception['bitwidth'] == 16 else 4
            quantizer.mantissa_bits = 10 if exception['bitwidth'] == 16 else 3
        else:
            quantizer.bitwidth = exception['bitwidth']

    if "asymmetric" in exception.keys():
        quantizer.symmetric = not exception["asymmetric"]
        quantizer.signed = not exception["asymmetric"]
        assert (quantizer.symmetric is True and quantizer.signed is True) or (quantizer.symmetric is False and quantizer.signed is False), "symmetric and signed must be aligned (True or False)"

    if "encoding_overrides" in exception.keys():
        encodings = exception['encoding_overrides']
        assert isinstance(encodings, dict)

        if not "is_symmetric" in encodings.keys():
            assert hasattr(quantizer, "_symmetric"), f"Quantizer {quantizer} doesn't have attribute \"_symmetric\""
            # set_legacy_encodings function expects the value of is_symmetric to be a string
            encodings["is_symmetric"] = str(quantizer._symmetric)

        if not "bitwidth" in encodings.keys():
            assert hasattr(quantizer, "bitwidth"), f"Quantizer {quantizer} doesn't have attribute \"bitwidth\""
            encodings["bitwidth"] = quantizer.bitwidth

        quantizer.set_legacy_encodings([encodings])
        quantizer._allow_overwrite = False

    return quantizer

def apply_param_exception(quantizer, exception):
    assert quantizer
    assert isinstance(exception, dict), f"Exception is not a dictionary type: {exception}"

    if exception.get("enabled", True) is False:
        quantizer._allow_overwrite = False

    quantizer.symmetric = exception.get("asymmetric", False) is False
    quantizer.signed = exception.get("asymmetric", False) is False
    assert (quantizer.symmetric is True and quantizer.signed is True) or (quantizer.symmetric is False and quantizer.signed is False), "symmetric and signed must be aligned (True or False)"
    quantizer.bitwidth = exception.get("bitwidth", 4)

    return quantizer

def get_module_exception(module, exception_dict):
    if type(module).__name__ in exception_dict.keys():
        return exception_dict[module._get_name()]

    return None


class ManualQuantsimMixedPrecisionConfig:
    def __init__(self, mixed_precision_config_file):

        assert isinstance(mixed_precision_config_file, (str,dict)), f"dtype {type(mixed_precision_config_file)} is not supported"
        exception_types = ("name", "module")
        exceptions = {k: defaultdict(lambda: nop_exception) for k in exception_types}


        if(isinstance(mixed_precision_config_file,str)):
            assert os.path.exists(mixed_precision_config_file),f"mixed precision file not found at given location: {mixed_precision_config_file}"
            with open(mixed_precision_config_file) as f:
                exception_config = json.load(f)
        else:
            exception_config=mixed_precision_config_file
        # Populate op_list here
        for etype in exception_types:
            for item in exception_config[f'{etype}_list']:
                expections_str = {k: v for k, v in item['exceptions'].items() if v is not None}
                print(f"Applying {item['module_name']}:\t{expections_str}")
                exceptions[etype].update({item['module_name']: item['exceptions']})
        self.exceptions_dict = exceptions

    def apply_exceptions(self, quant_sim):
        for etype in ("module", "name"):
            exception_modules = self.exceptions_dict[etype].keys()

            for name, module in quant_sim.model.named_modules():
                if isinstance(module, BaseQuantizationMixin):
                    exception = None
                    if etype == 'module':
                        exception = get_module_exception(module, self.exceptions_dict[etype])
                    elif etype == 'name':
                        for key in exception_modules:
                            if "*" in key and key.replace("*", "") in name:
                                exception = self.exceptions_dict[etype][key]
                            else:
                                match = re.fullmatch(key, name)
                                if match:
                                    exception = self.exceptions_dict[etype][key]

                    if exception is not None:
                        if exception["param_exceptions"] is not None:
                            self.apply_param_exception_to_module(exception["param_exceptions"], module)

                        if exception["input_exceptions"] is not None:
                            self.apply_input_exception_to_module(exception["input_exceptions"], module)

                        if exception["output_exceptions"] is not None:
                            self.apply_output_exception_to_module(exception["output_exceptions"], module)

    def apply_param_exception_to_module(self, param_exceptions, module):
        is_enable = (int(param_exceptions["bitwidth"]) < 32) if "bitwidth" in param_exceptions else True
        if not is_enable:
            param_exceptions["enabled"] = is_enable

        module.param_quantizers['weight'] = apply_param_exception(module.param_quantizers['weight'], param_exceptions)

    def apply_input_exception_to_module(self, input_exceptions, module):
        for index in range(len(input_exceptions)):
            module.input_quantizers[index] = apply_input_output_exception(module.input_quantizers[index], input_exceptions[index])

    def apply_output_exception_to_module(self, output_exceptions, module):
        for index in range(len(output_exceptions)):
            module.output_quantizers[index] = apply_input_output_exception(module.output_quantizers[index], output_exceptions[index])
