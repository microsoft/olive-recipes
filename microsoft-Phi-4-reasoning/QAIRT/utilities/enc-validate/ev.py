#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Dict, Iterable

import argparse
import collections
import datetime
import heapq
import html
import itertools
import json
import numpy as np
import os
#import pydot
import re
import sys
import pprint
import copy

import onnx
import onnx.numpy_helper as numpy_helper
from onnx.external_data_helper import uses_external_data

from onnx import (
    NodeProto,
    TensorProto,
    ValueInfoProto,
)

TensorInfo = collections.namedtuple('TensorInfo', 'name type shape array attrs')

def _br(): return "<br/>"
def _bold(s): return f"<b>{s}</b>"
def _red(s): return f'<font color="red">{s}</font>'
def _blue(s): return f'<font color="blue">{s}</font>'
def _esc(s): return html.escape(s)

def _f(f): return f'{f:.6}'
def _fclose(f1, f2, eps=1e-10): return (abs(abs(f1)-abs(f2))<eps)


def _match(pattern, txt):
    return re.match(f'.*{pattern}', txt) is not None

def _round(f): # Implement 'round half away from zero', C-style rounding
    from math import floor
    sign = -1 if f < 0 else 1
    return int(floor(abs(f) + 0.5) * sign)

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def load_aimet_encoding(encodingfiles, csv_info=None):
    def _load(encfile, csv_info):
        print('Loading', encfile)
        with open(encfile) as json_file:
            quant_encoding_dict = json.load(json_file)

        this = {}
        if isinstance(quant_encoding_dict['param_encodings'], dict):
            Encoding = AimetEncodingOld
            this.update({key: Encoding(enc, 'activation') for key, enc in quant_encoding_dict['activation_encodings'].items()})
            this.update({key: Encoding(enc, 'parameter') for key, enc in quant_encoding_dict['param_encodings'].items()})
            if csv_info is not None:
                csv_info.append(f"Number of Activation Encoding Summary, There are {len(quant_encoding_dict['activation_encodings'])} activation encodings in total\n")
                csv_info.append(f"Number of Param Encoding Summary, There are {len(quant_encoding_dict['param_encodings'])} param encodings in total\n")

        elif isinstance(quant_encoding_dict['param_encodings'], list):
            Encoding = AimetEncodingBq
            this.update({enc['name']: Encoding(enc, 'activation') for enc in quant_encoding_dict['activation_encodings']})
            this.update({enc['name']: Encoding(enc, 'parameter') for enc in quant_encoding_dict['param_encodings']})
            if csv_info is not None:
                csv_info.append(f"Number of Activation Encoding Summary, There are {len(quant_encoding_dict['activation_encodings'])} activation encodings in total\n")
                csv_info.append(f"Number of Param Encoding Summary, There are {len(quant_encoding_dict['param_encodings'])} param encodings in total\n")
        else:
            assert False, f'unknown type of AIMET encoding: {type(type(quant_encoding_dict["param_encodings"]))} from {encfile}'

        overlap = all.keys() & this.keys()
        if overlap:
            print(f'Encoding overlap: {overlap}')
        all.update(this)

    all = {}
    for encfile in encodingfiles:
        _load(encfile, csv_info)
    print(f"keys inside the encoding file is {len(all)}")
    return EncodingDict(all)

def load_json_map(map_path):
    map_dict = {}
    with open(map_path) as json_file:
        json_map = json.load(json_file)
    for mha_key, sha_keys in json_map["activation_encodings"].items():
        map_dict.update({mha_key: sha_keys})
    for mha_key, sha_keys in json_map["param_encodings"].items():
        map_dict.update({mha_key: sha_keys})
    return map_dict

def load_sha_aimet_encodings_and_maps(mha_encodings, sha_encoding_paths, mha2sha_map_paths, csv_info):
    map_dicts = []
    sha_dicts = []
    merged_sha_encodings = {}
    merged_map = {}

    tmp_mha_encodings = {}
    for key, value in mha_encodings.items():
        tmp_mha_encodings.update({key: value})

    for mha2sha_map in mha2sha_map_paths:
        map_dicts.append(load_json_map(mha2sha_map))
    for sha_encoding in sha_encoding_paths:
        tmp = []
        tmp.append(sha_encoding)
        sha_dicts.append(load_aimet_encoding(tmp))

    total_mha_keys = len(mha_encodings)

    total_sha_keys = 0
    for sha_dict in sha_dicts:
        total_sha_keys += len(sha_dict)

    if len(map_dicts) != len(sha_dicts):
        raise KeyError(f"Mismatch sha encoding files and map files, {len(sha_dicts)} sha encoding files and {len(map_dicts)} map files")

    sha_keys_num = 0
    duplicate_in_same_map = [0] * len(map_dicts)
    duplicate = {}
    for mha_key in tmp_mha_encodings:
        mha_key_in_map_count = 0
        tmp_list = []
        for idx, map_dict in enumerate(map_dicts):
            current_sha_dict = sha_dicts[idx]
            if mha_key in map_dict:
                sha_keys = map_dict[mha_key]
                sha_keys_num += len(sha_keys)
                if mha_key_in_map_count == 0:
                    new_sha_keys = []
                    for sha_key in sha_keys:
                        if sha_key in merged_sha_encodings and duplicate[sha_key] != idx:   
                            new_sha_key = f"split_{str(idx)}_" + sha_key
                            merged_sha_encodings.update({new_sha_key: current_sha_dict[sha_key]})
                            duplicate.update({new_sha_key: idx})
                            new_sha_keys.append(new_sha_key)
                        elif sha_key in merged_sha_encodings and duplicate[sha_key] == idx:
                            duplicate_in_same_map[idx] += 1
                            new_sha_keys.append(sha_key)
                        else:
                            merged_sha_encodings.update({sha_key: current_sha_dict[sha_key]})
                            duplicate.update({sha_key: idx})
                            new_sha_keys.append(sha_key)
                    merged_map.update({mha_key: new_sha_keys}) 
                else:
                    new_sha_keys = []
                    new_mha_key = f"split_{str(idx)}_" + mha_key
                    for sha_key in sha_keys:
                        new_sha_key = f"split_{str(idx)}_" + sha_key
                        if new_sha_key not in merged_sha_encodings: 
                            merged_sha_encodings.update({new_sha_key: current_sha_dict[sha_key]})
                            duplicate.update({new_sha_key: idx})
                        else:
                            duplicate_in_same_map[idx] += 1
                        new_sha_keys.append(new_sha_key)
                    merged_map.update({new_mha_key: new_sha_keys})
                    mha_encodings.update({new_mha_key: mha_encodings[mha_key]})
                mha_key_in_map_count += 1
        
        if mha_key_in_map_count > 1:
            csv_info.append(f"MHA Key overlap in Multiple Maps Info, {mha_key}, MHA key exists in {mha_key_in_map_count} MHA2SHA Maps\n")
        
        # One to one mapping
        found = False
        for sha_dict in sha_dicts:
            if mha_key in sha_dict:
                found = True
        if found:
            mha_key_in_sha_count = 0
            new_sha_keys = []
            for idx, sha_dict in enumerate(sha_dicts):
                if mha_key in sha_dict:
                    if mha_key_in_sha_count == 0:
                        new_sha_keys.append(mha_key)
                        if mha_key not in merged_sha_encodings:
                            merged_sha_encodings.update({mha_key: sha_dict[mha_key]})
                            duplicate[mha_key] = idx
                        else:
                            duplicate_in_same_map[idx] += 1
                    else:
                        new_sha_key = f"split_{str(idx)}_" + mha_key
                        new_mha_key = new_sha_key
                        if new_sha_key not in merged_sha_encodings:
                            merged_sha_encodings.update({new_sha_key: sha_dict[mha_key]})
                        else:
                            duplicate_in_same_map[idx] += 1 
                        mha_encodings.update({new_mha_key: mha_encodings[mha_key]})
                    mha_key_in_sha_count += 1
                    sha_keys_num += 1
            
            if mha_key_in_sha_count == 0:
                csv_info.append(f"MHA Key Miss Error, {mha_key}, MHA Key does not exist in any SHA encoding file\n")
            elif mha_key_in_sha_count > 1:
                merged_map.update({mha_key: new_sha_keys})
                csv_info.append(f"One-to-one map MHA Key overlap in Multiple SHA Info, {mha_key}, MHA key  exists in {mha_key_in_sha_count} SHA encoding files\n")
    total_mha_keys = len(mha_encodings)
    total_duplicate_keys = 0
    for i in duplicate_in_same_map:
        total_duplicate_keys += i
    if sha_keys_num - total_duplicate_keys != total_sha_keys:
        csv_info.append(f"MHA2SHA Key Total Number Mismatch Error, , By traversing {total_mha_keys} MHA Keys, we are able to access {sha_keys_num} SHA Keys however there are {total_sha_keys} SHA keys in total\n")
    if len(merged_sha_encodings) != total_sha_keys:
        print(f"Something is wrong, {len(merged_sha_encodings)} encodings in merged_sha_encodings, but {total_sha_keys} in total")

    return merged_sha_encodings, merged_map 

def get_activation_encodings(encodings):
    act_encs = {}
    for key, value in encodings.items():
        if value.type == 'activation':
            act_encs.update({key: value})
    return act_encs

def get_param_encodings(encodings):
    param_encs = {}
    for key, value in encodings.items():
        if value.type == 'parameter':
            param_encs.update({key: value})
    return param_encs

def compare_activation_values(act_encs1, act_encs2, csv_info):

    offset_counter = 0
    scale_counter = 0
    min_counter = 0
    max_counter = 0
    is_symmetric_counter = 0
    dtype_counter = 0
    bitwidth_counter = 0

    miss_key_1to2 = 0

    for key, value1 in act_encs1.items():
        value2 = act_encs2.get(key)
        if value2 is None:
            csv_info.append(f"Miss Activation Key Error, {key}, this tensor exists in encoding file 1 but does not exist in encoding file 2\n")
            print(f"Miss Key Error, {key}, this tensor exists in encoding file 1 but does not exist in encoding file 2\n")
            miss_key_1to2 += 1
            continue
        if isinstance(value1.encoding, list):
            for dict1, dict2 in zip(value1.encoding, value2.encoding):
                offset1 = dict1["offset"]
                offset2 = dict2["offset"]

                scale1 = dict1["scale"]
                scale2 = dict2["scale"]

                is_symmetric1 = dict1["is_symmetric"]
                is_symmetric2 = dict2["is_symmetric"]

                dtype1 = dict1["dtype"]
                dtype2 = dict2["dtype"]

                bitwidth1 = dict1["bitwidth"]
                bitwidth2 = dict2["bitwidth"]

                if offset1 != offset2:
                    offset_counter = offset_counter + 1
                
                if scale1 != scale2:
                    scale_counter = scale_counter + 1

                if is_symmetric1 != is_symmetric2:
                    is_symmetric_counter = is_symmetric_counter + 1

                if dtype1 != dtype2:
                    dtype_counter = dtype_counter + 1

                if bitwidth1 != bitwidth2:
                    bitwidth_counter = bitwidth_counter + 1
        elif isinstance(value1.encoding, dict):
            dict1 = value1.encoding
            dict2 = value2.encoding
            offset1 = dict1["offset"]
            offset2 = dict2["offset"]

            scale1 = dict1["scale"]
            scale2 = dict2["scale"]


            is_symmetric1 = dict1["is_sym"]
            is_symmetric2 = dict2["is_sym"]

            dtype1 = dict1["dtype"]
            dtype2 = dict2["dtype"]

            bitwidth1 = dict1["bw"]
            bitwidth2 = dict2["bw"]

            if offset1 != offset2:
                offset_counter = offset_counter + 1
                
            if scale1 != scale2:
                scale_counter = scale_counter + 1

            if is_symmetric1 != is_symmetric2:
                is_symmetric_counter = is_symmetric_counter + 1

            if dtype1 != dtype2:
                dtype_counter = dtype_counter + 1

            if bitwidth1 != bitwidth2:
                bitwidth_counter = bitwidth_counter + 1
            
    csv_info.append(f"Activation Encodings Compare Summary, ,There are {len(act_encs1)} and {len(act_encs2)} activation encodings in the encoding file 1 and 2.\n")
    csv_info.append(f"Activation Encodings Compare Summary, , There are {offset_counter} offsets, {scale_counter} scales, {dtype_counter} dtype {bitwidth_counter} bitwidth and {is_symmetric_counter} is_symmetric in encoding file 1 DO NOT match the ones in encoding file 2\n")
    if miss_key_1to2 != 0:
        csv_info.append(f"Miss Activation Key Summary, , The are {miss_key_1to2} keys exist in encoding file 1 but DO NOT exist in encoding file 2\n")
    else:
        csv_info.append("Miss Activation Key Summary, , Activation key matches between two encoding files\n")
    print(f"There are {offset_counter} offsets, {scale_counter} scales, {dtype_counter} dtype {bitwidth_counter} bitwidth and {is_symmetric_counter} is_symmetric in encoding file 1 DO NOT match the ones in encoding file 2\n")
    print(f"The are {miss_key_1to2} keys exist in encoding file 1 but DO NOT exist in encoding file 2\n")
    return csv_info

def compare_param_encodings(para_encs1, para_encs2, csv_info):
    base_weight = True
    lora_weight = True
    lora_bit = True
    counter = 0
    base_weight_mismatch = 0
    lora_weight_mismatch = 0
    lora_weight_match = 0
    lora_weight_bitwidth_err = 0

    for key, value1 in para_encs1.items():
        value2 = para_encs2.get(key)
        if value2 is None:
            csv_info.append(f"Parameter Encoding Missing Error in LoRA Encoding Validation, {key}, This key exists in encoding file 1 but does not exist in encoding file 2")
            continue

        counter_1 = 0
        base_tmp = True
        lora_tmp = True
        lora_bit_tmp = True

        if isinstance(value1.encoding, list):
            for dict1, dict2 in zip(value1.encoding, value2.encoding):  # Key missing in second JSON
                if "lora" not in key:
                    bitwidth1 = dict1["bitwidth"]
                    bitwidth2 = dict2["bitwidth"]

                    offset1 = dict1["offset"]
                    offset2 = dict2["offset"]

                    scale1 = dict1["scale"]
                    scale2 = dict2["scale"]

                    base_tmp = base_tmp & (bitwidth1 == bitwidth2) & (offset1 == offset2) & (scale1 == scale2)
                    counter_1 = counter_1 + 1       
                else:
                    scale1 = dict1["scale"]
                    scale2 = dict2["scale"]

                    bitwidth1 = dict1["bitwidth"]
                    bitwidth2 = dict2["bitwidth"]

                    lora_tmp = lora_tmp & (scale1 == scale2)
                    lora_bit_tmp = lora_bit_tmp & (bitwidth1 == 16) & (bitwidth2 == 16)
                    counter_1 = counter_1 + 1
        elif isinstance(value1.encoding, dict):
            dict1 = value1.encoding
            dict2 = value2.encoding
            if "lora" not in key:
                bitwidth1 = dict1["bw"]
                bitwidth2 = dict2["bw"]

                offset1 = dict1["offset"]
                offset2 = dict2["offset"]

                scale1 = dict1["scale"]
                scale2 = dict2["scale"]

                base_tmp = base_tmp & (bitwidth1 == bitwidth2) & (offset1 == offset2) & (scale1 == scale2)
            else:
                scale1 = dict1["scale"]
                scale2 = dict2["scale"]

                bitwidth1 = dict1["bw"]
                bitwidth2 = dict2["bw"]

                lora_tmp = lora_tmp & (scale1 == scale2)
                lora_bit_tmp = lora_bit_tmp & (bitwidth1 == 16) & (bitwidth2 == 16)
                
        if base_tmp == False:
            csv_info.append(f"Base Weights Mismatch Error, {key}, This encoding ({len(value1.encoding)} channels) in encoding file 1 does not match the one in encoding file 2\n")
            base_weight_mismatch += 1
            print(f"The encodings of base weight {key} ({len(value1.encoding)} channels) in encoding file 1 does not match the one in encoding file 2")
        if (lora_tmp == True) and ("lora" in key):
            lora_weight_match += 1
            print(f"The encodings of lora weight {key} ({len(value1.encoding)} channels) in encoding file 1 match the ones in encoding file 2")
        elif (lora_tmp == False) and ("lora" in key):
            lora_weight_mismatch += 1
            print(f"The encodings of lora weight {key} ({len(value1.encoding)} channels) in encoding file 1 DO NOT match the ones in encoding file 2")
        if (lora_bit_tmp == False):
            lora_weight_bitwidth_err += 1
            csv_info.append(f"LoRA Weights Encoding bitwidth Error, {key}, encoding file 1 is {bitwidth1} bit and {bitwidth2} in encoding file 2\n")
            print(f"The {key} of encoding file 1 is {bitwidth1} bit and {bitwidth2} in encoding file 2")
        if isinstance(value1.encoding, list):
            base_weight = base_weight & base_tmp & (counter_1 == len(value1.encoding)) & (counter_1 == len(value1.encoding))
            lora_weight = lora_weight & lora_tmp & (counter_1 == len(value1.encoding)) & (counter_1 == len(value1.encoding))
            lora_bit = lora_bit & lora_bit_tmp & (counter_1 == len(value1.encoding)) & (counter_1 == len(value1.encoding))
        elif isinstance(value1.encoding, dict):
            base_weight = base_weight & base_tmp
            lora_weight = lora_weight & lora_tmp
            lora_bit = lora_bit & lora_bit_tmp

        counter = counter + 1
    
    print(f"counter is {counter}")
    base_weight = base_weight & (counter == len(para_encs1)) & (counter == len(para_encs2))
    print(f"base_weight is {base_weight}")
    csv_info.append(f"Parameter Encodings Compare Summary, , There are {len(para_encs1)} and {len(para_encs2)} parameter encodings in the encoding file 1 and 2\n")
    csv_info.append(f"Mismatch Base Weight Encodings Summary, , There are {base_weight_mismatch} base weight encodings mismatch between two encoding files\n")
    csv_info.append(f"MisMatch LoRA Weight Encodings Summary, , There are {lora_weight_mismatch} lora weight encodings mismatch between two encoding files\n") 
    csv_info.append(f"Match LoRA Weight Encodings Summary, , There are {lora_weight_match} encodings match between two encoding files\n")
    csv_info.append(f"LoRA Weight encoding bitwidth mismatch, , There are {lora_weight_bitwidth_err} encodings bitwidth mismatch between two encoding files\n")     
    
    return base_weight, lora_bit, lora_weight, csv_info

def encoding_validation(encodings, sec_encodings, csv_info):
    act_encs1 = get_activation_encodings(encodings)
    print(type(sec_encodings))
    act_encs2 = get_activation_encodings(sec_encodings)
    para_encs1 = get_param_encodings(encodings)
    para_encs2 = get_param_encodings(sec_encodings)

    csv_info = compare_activation_values(act_encs1, act_encs2, csv_info)
    base_match, lora_bit, lora_match, csv_info = compare_param_encodings(para_encs1, para_encs2, csv_info)
        
    if base_match:
        csv_info.append(f"Base Weights Equal Info Summary, , The base weight of two adaptors are exactly the same !\n")
        print("The base weight of two adaptors are exactly the same !!!")
    else:
        csv_info.append(f"Base Weights Equal Info Summary, , The base weight of two adaptors are NOT the same !\n")
        print("The base weight of two adaptors are NOT the same !!!")

    if lora_bit:
        csv_info.append(f"LoRA Weights Bitwidth Summary, , The quantized lora weight is 16 bit, and lora weights are different between different adaptors !\n")
        print("The quantized lora weight is 16 bit, and lora weights are different between different adaptors !!!")
    else:
        csv_info.append(f"LoRA Weights Bitwidth Summary, , The quantized lora weight are not 16 bit or exactly the same !\n")
        print("The quantized lora weight are not 16 bit or exactly the same !!!")

    if lora_match:
        csv_info.append(f"LoRA Weights Equal Info Summary, , The lora weight of two adaptors are exactly the same !\n")
    else:
        csv_info.append(f"LoRA Weights Equal Info Summary, , The lora weight of two adaptors are Not the same !\n")

    return csv_info

def check_mha2sha_per_channel(mha_encoding, sha_encoding, mha_channel_offset):
    if mha_encoding["bw"] != sha_encoding["bw"]:
        return False
    if mha_encoding["dtype"] != sha_encoding["dtype"]:
        return False
    if mha_encoding["is_sym"] != sha_encoding["is_sym"]:
        return False
    for idx, (offset, scale) in enumerate(zip(sha_encoding["offset"], sha_encoding["scale"])):
        if mha_encoding["offset"][idx+mha_channel_offset] != offset:
            return False
        if mha_encoding["scale"][idx+mha_channel_offset] != scale:
            return False
    return True

def check_mha2sha_equal(mha_encoding, sha_encoding):
    if mha_encoding["bw"] != sha_encoding["bw"]:
        return False
    if mha_encoding["dtype"] != sha_encoding["dtype"]:
        return False
    if mha_encoding["is_sym"] != sha_encoding["is_sym"]:
        return False
    if mha_encoding["offset"] != sha_encoding["offset"]:
        return False
    if mha_encoding["scale"] != sha_encoding["scale"]:
        return False
    return True

def mha2sha_compare(enc_map, mha_enc, sha_enc, csv_info):

    validate = True
    pt_pc_mism = {}
    enc_mism = {}
    total_channel_mism = []
    only_mha_miss = []
    only_sha_miss = []
    both_miss = {}
    map_miss = []

    count = 0
    tmp_count = 0

    def insert_key_to_dict(mha_key, sha_key, dst_dict):
        if mha_key not in dst_dict:
            dst_dict.update({mha_key: []})
            dst_dict[mha_key].append(sha_key)
        else:
            dst_dict[mha_key].append(sha_key)

    for mha_key, sha_keys in enc_map.items():
        mha_miss = False
        sha_miss = False
        if mha_key not in mha_enc:
            mha_miss =True
#            print(f"{mha_key} exist in mha_to_sha_map but does not exist in mha encodings")
        
        miss_sha_keys = []
        for sha_key in sha_keys:
            if sha_key not in sha_enc:
                sha_miss = True
                miss_sha_keys.append(sha_key)
        if mha_miss and (not sha_miss):
            only_mha_miss.append(mha_key)
            continue
        elif (not mha_miss) and sha_miss:
            only_sha_miss.append(sha_key)
            continue
        elif mha_miss and sha_miss:
            both_miss.update({mha_key: miss_sha_keys})
            continue
    
    for mha_key, mha_encodings in mha_enc.items():
        if mha_key in sha_enc:
            count += 1
            sha_encodings = sha_enc[mha_key].encoding
            if isinstance(sha_encodings, list):
                if len(mha_encodings.encoding) != len(sha_encodings):
                    insert_key_to_dict(mha_key, mha_key, pt_pc_mism)
                elif mha_encodings.encoding != sha_encodings:
                    validate = False
                    insert_key_to_dict(mha_key, mha_key, enc_mism)
                else:
                    tmp_count += 1
            elif isinstance(sha_encodings, dict):
                if mha_encodings.encoding["enc_type"] != sha_encodings["enc_type"]:
                    insert_key_to_dict(mha_key, mha_key, pt_pc_mism)
                elif not check_mha2sha_equal(mha_encodings.encoding, sha_encodings):
                    validate = False
                    insert_key_to_dict(mha_key, mha_key, enc_mism)
                else:
                    tmp_count += 1
            else:
                raise ValueError("Unsupported Encoding Structure")               
        if mha_key in enc_map:
            mha_list = mha_enc[mha_key].encoding
            sha_keys = enc_map[mha_key]
            if isinstance(mha_list, list):
                # Per tensor encoding:
                if len(mha_list) == 1:
                    mha_encoding = mha_list[0]
                    for sha_key in sha_keys:
                        count += 1
                        sha_list = sha_enc[sha_key].encoding
                        sha_encoding = sha_list[0]
                        if len(sha_list) != 1:
                            validate = False
                            insert_key_to_dict(mha_key, sha_key, pt_pc_mism)
                        elif (len(sha_list) == 1) and (sha_encoding != mha_encoding):
                            validate = False
                            insert_key_to_dict(mha_key, sha_key, enc_mism)
                        else:
                            tmp_count += 1
                # Per channel encoding:
                else:
                    mha_channels = len(mha_list)
                    sha_channels = 0
                    for sha_key in sha_keys:
                        count += 1
                        sha_list = sha_enc[sha_key].encoding
                        tmp_tag = True
                        for idx in range(len(sha_list)):
                            mha_encoding = mha_list[sha_channels + idx]
                            sha_encoding = sha_list[idx]
                            if sha_encoding != mha_encoding:
                                validate = False
                                insert_key_to_dict(mha_key, sha_key, enc_mism)
                                tmp_tag = False
                                break
                        if tmp_tag:
                            tmp_count += 1
                        sha_channels += len(sha_list)
                    if mha_channels != sha_channels:
                        validate = False
                        total_channel_mism.append((mha_key, mha_channels, sha_channels))
            elif isinstance(mha_list, dict):
                mha_encoding = mha_list
                if mha_list["enc_type"] == 'PER_TENSOR':
                    for sha_key in sha_keys:
                        count += 1
                        sha_encoding = sha_enc[sha_key].encoding
                        if sha_encoding["enc_type"] != mha_list["enc_type"]:
                            validate = False
                            insert_key_to_dict(mha_key, sha_key, pt_pc_mism)
                        elif (sha_encoding["enc_type"] == mha_list["enc_type"]) and not check_mha2sha_equal(mha_encoding, sha_encoding):
                            validate = False
                            insert_key_to_dict(mha_key, sha_key, enc_mism)
                        else:
                            tmp_count += 1
                elif mha_list["enc_type"] == 'PER_CHANNEL':
                    mha_channels = len(mha_encoding["offset"])
                    mha_channel_offset = 0
                    for sha_key in sha_keys:
                        count += 1
                        sha_encoding = sha_enc[sha_key].encoding
                        if not check_mha2sha_per_channel(mha_encoding, sha_encoding, mha_channel_offset):
                            validate = False
                            insert_key_to_dict(mha_key, sha_key, enc_mism)
                        else:
                            tmp_count += 1
                        mha_channel_offset += len(sha_encoding["offset"])
                    if mha_channels != mha_channel_offset:
                        validate = False
                        total_channel_mism.append((mha_key, mha_channels, mha_channel_offset))
            else:
                raise ValueError("Unsupported Encoding Structure")
        
        if mha_key not in sha_enc and mha_key not in enc_map:
            map_miss.append(mha_key)

    print(f"validate is {validate}, count is {count}, tmp_count is {tmp_count}, len of map_miss is {len(map_miss)}, len_of pt_pc_mism is {len(pt_pc_mism)}, len of enc_mism is {len(enc_mism)}")
    validate = validate & (count == tmp_count) & (len(map_miss) == 0) & (not bool(pt_pc_mism)) & (not bool(enc_mism))

    csv_info.append(f"MHA2SHA Encoding validation Summary, , There are {len(mha_enc)} and {len(sha_enc)} encodings in MHA and SHA encodings\n")
    print(f"================== The compare between MHA and SHA encoding ================")
    if bool(pt_pc_mism):
        print("The items in per-tensor per-channel mismatch dictionary contains the encodings whose MHA is per tensor but SHA is per channel")
        for key, value in pt_pc_mism.items():
            csv_info.append(f"Per-Tensor, Per-Channel Encoding Mismatch Error, MHA: {key}/SHA: {value}, Encoding type mismatch\n")
        csv_info.append(f"Per-Tensor, Per-Channel Encoding Mismatch Summary, , There are {len(pt_pc_mism)} mismatches")
        print(pprint.pformat(pt_pc_mism))
    if bool(enc_mism):

        for key, value in enc_mism.items():
            csv_info.append(f"MHA SHA Encoding Mismatch Error, MHA: {key}/SHA: {value}, Encoding mismatch\n")
        csv_info.append(f"MHA SHA Encoding Mismatch Summary, , There are {len(enc_mism)} encoding mismatch\n")
        print("The items in encoding mismatch dictionary contains the encodings whose SHA encodings do not match the MHA one")
        print(pprint.pformat(enc_mism))

    if total_channel_mism:
        print("The items in this list contains the encodings whose total numbers of MHA encoding does not equal to the total channel of all the corresponding SHA encodings")
        for item in total_channel_mism:
            print(item)
    if only_mha_miss:
        print("The items in this list contains MHA encodings that exist in mha_to_sha map, but do not exist in the MHA encodings. The corresponding SHA encoding exist in the SHA encodings")
        for item in only_mha_miss:
            print(item)
    if only_sha_miss:
        print("The items in this list contains SHA encodings that exist in mha_to_sha map, but do not exist in the SHA encodings. The corresponding MHA encoding exist in the MHA encodings")
        for item in only_sha_miss:
            print(item)
    if bool(both_miss):
        print("The key exists in mha_to_sha map, but does not exist in MHA and SHA encodings")
        print(pprint.pformat(both_miss))
    if map_miss:
        for key in map_miss:
            csv_info.append(f"SHA Encoding Miss Error, {key}, This encoding exists in MHA but not SHA\n")
        csv_info.append(f"SHA Encoding Miss Summary, , There are {len(map_miss)} encoding exists in MHA but not SHA\n")
        
    print(f"validate is {validate}")
    return validate, csv_info

def validate_mha_sha_encodings(encodings, sec_encodings, map_dict, csv_info):
    validate, csv_info = mha2sha_compare(map_dict, encodings, sec_encodings, csv_info)

    if validate:
        csv_info.append("MHA2SHA Validation Summary, , Encodings pass the mha to sha validation !\n")
        print("Encodings pass the mha to sha validation")
    else:
        csv_info.append("MHA2SHA Encoding Validation Summary, , Encodings does not pass the mha to sha validation !\n")
        print("Encodings does not pass the mha to sha validation")
    return csv_info

class Encoding:
    def __init__(self, enc_type=None):
        self.count = 0
        self.bitwidth = 0
        self.dtype = ''
        self.max = ''
        self.min = ''
        self.offset = ''
        self.scale = ''
        self.type = enc_type

    def check(self, bitwidth=None, symmetric=None, scale=None, offset=None):
        if bitwidth is not None:
            if isinstance(bitwidth, Iterable) and self.bitwidth not in bitwidth: return False
#            print(f"self.bitwidth = {self.bitwidth}, bitwidth = {bitwidth}")
            if self.bitwidth != bitwidth: return False
        if symmetric is not None:
            if self.symmetric != symmetric: return False
        if scale is not None:
#            print(f"self.scale = {self.scale}, scale = {scale}")
            if not self.scale: return False
            if isinstance(scale, Iterable) and self.scale not in scale: return False
            if not _fclose(self.scale, scale): return False
        if offset is not None:
            if isinstance(offset, Iterable) and self.offset not in offset: return False
            if self.offset != offset: return False
        return True

    def __getattr__(self, name):
        return 'None'

class AimetEncodingOld(Encoding):
    def __init__(self, encoding, enc_type=None):
        super().__init__(enc_type)
        self.encoding = encoding
        self.count = len(encoding)
        self.bitwidth = encoding[0]['bitwidth']
        self.dtype = encoding[0]['dtype']
        self.max = encoding[0]['max']
        self.min = encoding[0]['min']
        self.offset = encoding[0]['offset']
        self.scale = encoding[0]['scale']

    @property
    def symmetric(self):
        return self.encoding[0]['is_symmetric'].lower() == 'true'

    def __repr__(self):
        return f'AIMET:q:{_f(self.scale)}:{self.offset}' \
            f' bw:{self.bitwidth}'                      \
            f' {_f(self.min)}~{_f(self.max)}'           \
            f' {" symmetric" if self.symmetric else ""}'\
            f' {f" ...[{self.count }]" if self.count > 1 else ""}'

class AimetEncodingBq(Encoding):
    def __init__(self, encoding, enc_type=None):
        super().__init__(enc_type)
        self.encoding = encoding
        self.count = len(encoding['offset']) if 'offset' in encoding and isinstance(encoding['offset'], list) else 1
        self.bitwidth = encoding['bw']
        self.dtype = encoding['dtype']
        self.enc_type = encoding['enc_type']
        self.block_size = encoding['block_size'] if 'block_size' in encoding else None
        self.compressed_bw = encoding['compressed_bw'] if 'compressed_bw' in encoding else None
        self.offset = encoding['offset'][0] if 'offset' in encoding else None
        self.scale = encoding['scale'][0] if 'scale' in encoding else None
        self.per_block_int_scale = encoding['per_block_int_scale'][0] if 'per_block_int_scale' in encoding else None

    @property
    def symmetric(self):
        return self.encoding.get('is_sym', False)

    def __repr__(self):
        fmt = f'AIMET:'
        if self.dtype.lower() == "int":
            ret_str = f'{fmt} q[0]:{_f(self.scale)}:{self.offset}' \
                f' bw:{self.bitwidth}'                      \
                f' {" symmetric" if self.symmetric else ""}'
            if self.enc_type == "PER_BLOCK":
                ret_str += f'{self.enc_type} {f" ...[n_channels_out*n_blocks:{self.count }]" if self.count > 1 else ""}'
            if self.enc_type == "LPBQ":
                ret_str += f'{self.enc_type} {f" ...[n_channels_out:{self.count }]" if self.count > 1 else ""}'
                ret_str += f' {f"compressed_bw: {self.compressed_bw}" if self.compressed_bw else "" }'\
                f' {f"int_scale[0]: {self.per_block_int_scale}" if self.per_block_int_scale else "" }'
            if self.enc_type =="BQ" or self.enc_type == "PER_BLOCK":
                ret_str += f'{self.enc_type} {f"block_size: {self.block_size}" if self.block_size else "" }'
            return ret_str

        elif self.dtype.lower() == "float":
            return f'{fmt} float bw:{self.bitwidth}'
        else:
            assert False, "Unexpected AIMET dtype:{self.dtype}"

class WordTrie():
    seperator = '._/'
    def __init__(self):
        self.children = {}
        self.name = None
        self.end = False

    @staticmethod
    def from_strings(strings, sep=seperator):
        root = WordTrie()
        for string in strings:
            words = re.split(f'[{sep}]', string)
            words = [word for word in words if word != '']
            root._add(string, words, 0)
        return root

    def match(self, string, sep=seperator):
        words = re.split(f'[{sep}]', string)
        return self._get(words, 0)

    def _add(self, fullname, words, offset):
        if offset == len(words):
            self.end = True
            self.name = fullname
        else:
            if words[offset] not in self.children:
                self.children[words[offset]] = WordTrie()
            self.children[words[offset]]._add(fullname, words, offset+1)

    def _get(self, words, offset):
        if offset == len(words): # complete match
            return True, self
        if words[offset] in self.children:
            return self.children[words[offset]]._get(words, offset+1)
        # partial match
        return False, self


class EncodingDict(collections.abc.MutableMapping):
    def __init__(self, encodings):
        self.encodings = encodings
        self.trie = WordTrie.from_strings(list(encodings.keys()))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, key):
        try:
            return self.encodings[key]
        except KeyError:
            pass

        matched, trie_node = self.trie.match(key)
        if matched:
            if trie_node.name:
                assert trie_node.name in self.encodings, f'Matched but no encoding[{trie_node.name}]'
                return self.encodings[trie_node.name]
        #else: # allow partial matching
        #    if trie_node.end and trie_node.name in self.encodings:
        #        print(f'Map encoding {trie_node.name} to {key}')
        #        return self.encodings[trie_node.name]
        raise KeyError(key)

    def __iter__(self):
        return iter(self.encodings)

    def __setitem__(self, key, value):
        self.encodings[key] = value

    def __delitem__(self, key):
        del self.encodings[key]

class OnnxModel():
    def __init__(self, onnxfile):
        self.onnxfile = onnxfile
        self.onnxmodel = onnx.load(onnxfile, load_external_data=False)

    def __getattr__(self, name): # behave like onnx models
        return self.onnxmodel.__getattribute__(name)

def _attr_label(attr):
    if attr.type == 1:  # float
        return f'{attr.f}'
    elif attr.type == 2:  # int
        return f'{attr.i}'
    elif attr.type == 3:  # string
        return f'{attr.s.decode("utf-8")}'.replace('->', '=')
    elif attr.type == 4:  # tensor
        a = numpy_helper.to_array(attr.t)
        return f'{a}' if a.size < 10 else f'{list(a.shape)}{list(a.reshape(-1)[:10])}...'
    elif attr.type == 5:  # graph
        return f'{attr.g.name}'
    elif attr.type == 6:  # floats
        return f'{tuple(attr.floats)}'
    elif attr.type == 7:  # ints
        return f'{tuple(attr.ints)}'
    elif attr.type == 8:  # strings
        return f'{[_esc(attr.decode("utf-8")) for attr in attr.strings]}'
    elif attr.type == 9:  # tensors
        return 'tensors'
    elif attr.type == 10:  # graphs
        assert False, 'Nog implemented'
        return f'{[g.name for g in attr.graphs]}'

def _tensor_data(arr):
    def _str(i):
        if isinstance(i, (np.float64,np.float32)):
            return f'{i:.3f}'
        return str(i)
    arr = arr.reshape(-1);
    data = [_str(i) for i in arr[:min(arr.size, 4)]]
    if arr.size > 4:
        data += ['...']
        data += [_str(i) for i in arr[max(arr.size-4, 4):]]
    return '[' + ','.join(data) + ']'


class DotNode:
    def __init__(self):
        self._inputs = []
        self._outputs = []

    @property
    def title(self) -> str:
        title = f'{self.name} [{self.type}]' if self.name else f'{self.type}'
        if self.node_index is not None:
            title += f'[{self.node_index}]'
        return title

    def get_tensor(self, is_input, index):
        return self._inputs[index] if is_input else self._outputs[index]


class OnnxNode(DotNode):
    def __init__(self, node, node_index: int):
        super().__init__()
        self.node = node
        self.type = node.op_type
        self.node_index = node_index
        self.notes = []
        self._inputs = list(node.input)
        self._outputs = list(node.output)
        self._attributes = {attr.name: _attr_label(attr) for attr in node.attribute}

    def __getattr__(self, name): # behave like onnx nodes
        return self.node.__getattribute__(name)

class DotGraph:
    def __init__(self, encodings):
        # nodes = {node.title: node}
        self.nodes = {} # Dict[str, NodeProto]
        # edges = [(src node_id, dst node_id)]
        self.edges = [] # List[str]
        # encoding = {tensor: [encoding]}
        self.encodings = encodings
        # constants = {tensor: Any}
        self.constants = {}

    def _build_graph(self, nodes: Dict[str, object]):
        edges = []
        up, down = collections.defaultdict(list), collections.defaultdict(list)
        consumers = collections.defaultdict(list)
        producer = {tensor: node_name for node_name, node in nodes.items() for tensor in node._outputs if len(tensor) > 0}
        for node_name, node in  nodes.items():
            for input_tensor in node._inputs:
                consumers[input_tensor].append(node_name)
                if input_tensor in producer:
                    src, dst = producer[input_tensor], node_name
                    edges.append((src, dst))
                    up[dst].append(src)
                    down[src].append(dst)
        self.nodes = nodes
        self.edges = edges
        self.adjacency_upward = up
        self.adjacency_downward = down
        self.producer = producer
        self.consumers = consumers

        def _topo_sort(root_nodes, successor, predecessor):
            result, visited = [], set()
            no_incoming_edge = collections.deque(root_nodes)
            while no_incoming_edge:
                cur = no_incoming_edge.popleft()
                if cur in visited: continue
                visited.add(cur)
                result.append(cur)
                for child in successor[cur]:
                    in_degree = len([pred for pred in predecessor[child] if pred not in visited])
                    if in_degree == 0:
                        no_incoming_edge.append(child)
            if len(result) != len(nodes):
                print("Error in topo_sort, missed these nodes:", [node for node in nodes not in visited])
            return result

        # input/output leaf nodes
        leaf_outputs = [producer[tensor] for tensor in self._outputs if len(down[producer[tensor]]) == 0 ]
        topo_sorted = _topo_sort(leaf_outputs, up, down)
        max_level, level = 0, {}
        for node in reversed(topo_sorted):
            level[node] = max([0] + [level[prev]+1 for prev in up[node] if prev in level])
            max_level = max(max_level, level[node])
        self.level = level
        self.max_level = max_level

    def _find_encoding(self, tensor):
        while tensor not in self.encodings:
            tensor = self.nodes[self.producer[tensor]]._inputs[0]
        return self.encodings[tensor]

    def _is_input(self, tensor: str) -> bool:
        return tensor in self._inputs

    def _is_output(self, tensor: str) -> bool:
        return tensor in self._outputs

    def _is_constant(self, tensor: str) -> bool:
        return tensor in self.constants

    def mismatch_dump(self, tensor, csv_info):
        return csv_info

def qnn_encoding_checker(self):

    def _check_tensor(bitwidth=None, symmetric=None, scale=None, offset=None):
            def _check(tensor):
                encoding = self._find_encoding(tensor)
                if encoding is not None:
                    if encoding.dtype.lower() == "float": 
                        return True 
                    return encoding.check(bitwidth=bitwidth, symmetric=symmetric, scale=scale, offset=offset)
                return False
            return _check

    def _check_tensor_dtype(dtype=None, bitwidth=None, symmetric=None, scale=None, offset=None):
        def _check(tensor):
            encoding = self._find_encoding(tensor)
            if encoding is not None:
                if encoding.dtype == '':
                    return True
                elif encoding.dtype.lower() == dtype and encoding.check(bitwidth=bitwidth, symmetric=symmetric, scale=scale, offset=offset):
                    return True 
            return False
        return _check

    def _custom(is_input=True, index=0, bitwidth=None, symmetric=None, scale=None, offset=None):
        def _check(node):
            tensor = node.get_tensor(is_input, index)
            encoding = self._find_encoding(tensor)
            if encoding:
                return encoding.check(bitwidth=bitwidth, symmetric=symmetric, scale=scale, offset=offset)
            return False
        return _check

    def _check_dtype(is_input=True, index=0, dtype=None, bitwidth=None, symmetric=None, scale=None, offset=None):
        def _check(node):
            tensor = node.get_tensor(is_input, index)
            encoding = self._find_encoding(tensor)
            if encoding is not None:
                if encoding.dtype == '':
                    return True
                elif encoding.dtype.lower() == dtype and encoding.check(bitwidth=bitwidth, symmetric=symmetric, scale=scale, offset=offset):
                    return True
            return False
        return _check

    def _symmetric(is_input, index):
        def _check(node):
            tensor = node.get_tensor(is_input, index)
            encoding = self._find_encoding(tensor)
            if encoding:
                return encoding.symmetric
            return False
        return _check

    def _aligned_with_input(input_indices):
            def _check(node):
                encodings = [self._find_encoding(node.get_tensor(is_input=False, index=0))]
                encodings += [self._find_encoding(node.get_tensor(is_input=True, index=i)) for i in input_indices]
                params = set((i.scale, i.offset) for i in encodings if i is not None)
                return len(params) <= 1
            return _check

    def _aligned_in_out(node):
        input_indices = list(range(len(node._inputs)))
        return _aligned_with_input(input_indices)(node)
    
    def _not_too_small_scale(encoding, eps=1e-10):
        if encoding.dtype.lower() == "float": 
            return True 
        return encoding.scale > eps if encoding.scale else True

    def not_too_large_scale(encoding, eps=1e+10):
        if encoding.dtype.lower() == "float": 
            return True 
        return encoding.scale < eps if encoding.scale else True

    def _large_max_detection(encoding, threshold=1000.0):
        if encoding.max == '':
            return False
        return encoding.max > threshold
    
    def _small_min_detection(encoding, threshold=-1000.0):
        if encoding.min == '':
            return False
        return encoding.min < threshold

    def _is_sigmoid(node):
        return node.type == 'ElementWiseNeuron' \
            and 'operation' in node._attributes and node._attributes['operation'] == 6
    
    def _non_lm_head_base_layer(node):
        input_name = node.get_tensor(is_input=True, index=1)
        return "lm_head" not in input_name and "lora" not in input_name

    def _is_lm_head_base_layer(node): 
        input_name = node.get_tensor(is_input=True, index=1)
        return "lm_head" in input_name

    def _is_base_layer(node):
        input_name = node.get_tensor(is_input=True, index=1)
        return "lora" not in input_name

    def _is_lora_layer(node):
        input_name = node.get_tensor(is_input=True, index=1)
        return "lora" in input_name

    def _is_reshape(node):
        output_name = node.get_tensor(is_input=False, index=0)
        return "Reshape_output_" in output_name

    def _is_transpose(node):
        output_name = node.get_tensor(is_input=False, index=0)
        return "Transpose_output_" in output_name

    def _is_cast(node):
        output_name = node.get_tensor(is_input=False, index=0)
        return "Cast" in output_name

    def _custom_op(_op_checker, is_input=True, index=0, bitwidth=None, symmetric=None, scale=None, offset=None):
        def _check(node):
            if _op_checker(node):
                checker = _custom(is_input=is_input, index=index, bitwidth=bitwidth, symmetric=symmetric, scale=scale, offset=offset)
                return checker(node)
            return True
        return _check

    def _custom_dtype_op(_op_checker, is_input=True, index=0, dtype=None, bitwidth=None, symmetric=None, scale=None, offset=None):
        def _check(node):
            if _op_checker(node):
                checker = _check_dtype(is_input=is_input, index=index, dtype=dtype, bitwidth=bitwidth, symmetric=symmetric, scale=scale, offset=offset)
                return checker(node)
            return True
        return _check
    
    def _check_op_exist(_op_checker):
        def _check(node):
            if _op_checker(node):
                tensor = node.get_tensor(is_input=False, index=0)
                if tensor in self.encodings:
                    return True
            return False
        return _check

    def _check_symmetric_offset(encoding):
        if encoding.dtype.lower() == "float": 
            return True 
        if encoding.symmetric \
            and encoding.offset != _qnn_symmetric_offset(encoding.bitwidth):
            return False
        return True

    def _not_too_large_scale(is_input, index=0, betaval=1.0):
        def _check(node):
            tensor = node.get_tensor(is_input, index)
            print(f"Checking the softmax input whose name is {tensor}")
            encoding = self._find_encoding(tensor)
            if encoding.scale is not None: 
                if encoding.dtype.lower() == "float": 
                    return True 
                scalep = encoding.scale * 1.4426950408889634 * betaval
                if encoding.bitwidth == 8:
                    return scalep < 128.0
                if encoding.bitwidth == 16:
                    return scalep < 1.0
                return True 
            return True 
        return _check

    rules_by_node = {
        'Gather': [
            (_aligned_with_input(input_indices=(0,)), "Input and output encodings should be equal"),
            (_custom(is_input=True, index=0, bitwidth=16), "Unexpected encoding setting"),
        ],
        'Transpose': [
            (_aligned_in_out, "input and output encodings should be equal"),
        ],
        'Reshape': [
            (_aligned_in_out, "input and output encodings should be equal"),
        ],
        'MatMul': [
            (_symmetric(is_input=True, index=1), "2nd input should be symmetric"),
        ],
        'Concat': [
            (_aligned_in_out, "input and output encodings should be equal"),
        ],
        'StridedSlice': [
            (_aligned_in_out, "input and output encodings should be equal"),
        ],
        'ElementWiseNeuron': [
            (_custom_op(_is_sigmoid, is_input=False, index=0, bitwidth=16, scale=1.52590e-5, offset=0), "Encoding shoule be scale:1.5259e-05 offset:0 bw:16"),
        ],
        'Conv': [
            (_symmetric(is_input=True, index=1), "2nd input should be symmetric"),
        ]
    }

#    node_type_error = {
#        'Gather': 0
#        'Transpose': 0
#        'Reshape': 0
#        'MatMul': 0
#        'Concat': 0
#        'StridedSlice': 0
#        'Sigmoid': 0
#        'Softmax': 0
#        'ElementWiseNeuron': 0
#        'Conv': 0
#    }

    rules_by_tensor = [  
        ('past_key_', _check_tensor(symmetric=True), "Key cache should have symmetric encodings"),
        ('past_value_', _check_tensor(symmetric=True), "Value cache should have symmetric encodings")
    ]

    rules_for_all = [
        (_not_too_small_scale, "Too small scale"),
        (not_too_large_scale, "Too large scale"),
        (_check_symmetric_offset, "Symmetric offset should be -2**(bitwidth-1)"),
    ]

    rules_by_node_exist = {
        'Transpose': [
            (_check_op_exist(_is_transpose), "Transpose exists inside the encoding file"),
        ],
        'Reshape': [
            (_check_op_exist(_is_reshape), "Reshape exists inside the encoding file"),
        ],
        'Cast': [
            (_check_op_exist(_is_cast), "Cast exists inside the encoding file"),  
        ]
    }

    rules_for_all_warning = {
        (_large_max_detection, "The maximum value of this encoding is larger than 1000 "),
        (_small_min_detection, "The maximum value of this encoding is smaller than -1000 ")
    }

#    node_exist_warn = {
#        'Transpose': 0
#        'Reshape': 0
#    }
    if "llm-bq" not in self.model_type:
        rules_by_node.update({'MatMul': [(_symmetric(is_input=True, index=1), "2nd input should be symmetric")]})
        rules_by_node.update({'Sigmoid': [(_custom(is_input=False, index=0, bitwidth=16, scale=1.52590e-5, offset=0), "Encoding shoule be scale:1.5259e-05 offset:0 bw:16")]})
        rules_by_node.update({'Softmax': [(_custom(is_input=False, index=0, bitwidth=16, scale=1.52590e-5, offset=0), "Encoding shoule be scale:1.5259e-05 offset:0 bw:16"),
                                          (_not_too_large_scale(is_input=True, betaval=1.0), "Softmax input encoding is too large to support integer softmax implementation")
                                         ]})
    if "lvm" in self.model_type:
        rules_by_node['Conv'].append((_custom_op(_is_base_layer, is_input=True, index=1, bitwidth=8), "The Conv weight parameters of Base Layer should be 8 bit in LVM model"))
    if "llm" in self.model_type:
        rules_by_node['Conv'].append((_custom_op(_non_lm_head_base_layer, is_input=True, index=1, bitwidth=4), "The non-lm_head Conv weight parameters of Base Layer should be 4 bit in LLM model"))
        rules_by_node['Conv'].append((_custom_op(_is_lm_head_base_layer, is_input=True, index=1, bitwidth=8), "The lm_head Conv weight parameters of Base Layer should be 8 bit in LLM model"))
        rules_by_tensor.append(('past_key_', _check_tensor_dtype(dtype="int", bitwidth=8), "Key cache should have 8 bit int encodings"))
        rules_by_tensor.append(('past_value_', _check_tensor_dtype(dtype="int", bitwidth=8), "Value cache should have 8 bit int encodings"))
    if "llm-bq" in self.model_type:
        rules_by_node.update({'MatMul': [(_check_dtype(is_input=True, index=1, dtype="float", bitwidth=16), "Second input of MatMul should be 16 bit float in BQ model")]})
        rules_by_node['Conv'].append((_custom_dtype_op(_is_base_layer, is_input=True, index=1, dtype="int", bitwidth=4), "Second input of Conv should be 4 bit int in BQ model"))
        rules_by_tensor.append(('past_key_', _check_tensor_dtype(dtype="float", bitwidth=16), "Key cache should have 16 bit float encodings in BQ"))
        rules_by_tensor.append(('past_value_', _check_tensor_dtype(dtype="float", bitwidth=16), "Value cache should have 16 bit float encodings in BQ"))
    if "llm-lpbq" in self.model_type:
        rules_by_node['Conv'].append((_custom_dtype_op(_is_base_layer, is_input=True, index=1, dtype="int", bitwidth=8), "Second input of Conv should be 8 bit int in LPBQ model"))
        rules_by_tensor.append(('past_key_', _check_tensor_dtype(dtype="int", bitwidth=8), "Key cache should have 8 bit int encodings"))
        rules_by_tensor.append(('past_value_', _check_tensor_dtype(dtype="int", bitwidth=8), "Value cache should have 8 bit int encodings"))
    if "lora" in self.model_type:
        rules_by_node['Conv'].append((_custom_op(_is_lora_layer, is_input=True, index=1, bitwidth=16), "The Conv weight parameters of LoRA Layer should be 16 bit in LoRA model"))

    csv_info = []
    csv_info.append("General Validation Info, Tensor Name, Detailed Log Message\n")

    for name, encoding in self.encodings.items():
        csv_info = self.mismatch_dump(name, csv_info)

    for name, node in self.nodes.items():
        rules = rules_by_node.get(node.type, None)
        if rules is not None:
            for checker, note in rules:
                try:
                    if not checker(node):
                        node.notes.append(note)
                        csv_info.append(f"Encoding Error that vialates rules for node, {node.name}, {note}\n")
                        print(f"Encoding error, {node.name}, {note}")
                except KeyError as e:
                    node.notes.append(f"KeyError {e}")
                except RuntimeError as e:
                    node.notes.append(f"RuntimeError {e}")

    for name, encoding in self.encodings.items():
        if name in self.producer:
            for pattern, checker, note in rules_by_tensor:
                if _match(pattern, name) and not checker(name):
                    self.nodes[self.producer[name]].notes.append(note)
                    csv_info.append(f"Encoding Error that vialates Rules for Tensor, {name}, {note} {encoding.dtype} {encoding.bitwidth}\n")
                    print(f"Encoding error, {name}, {encoding}, {note}")

        if name in self.producer:
            for checker, note in rules_for_all:
                if not checker(encoding):
                    self.nodes[self.producer[name]].notes.append(note)
                    csv_info.append(f"Encoding Error that vialates Rules for All, {name}, {note}, {encoding}\n")
                    print(f"Encoding error, {name}, {encoding}, {note}")

        if name in self.producer:
            for checker, note in rules_for_all_warning:
                if checker(encoding):
                    self.nodes[self.producer[name]].notes.append(note)
                    csv_info.append(f"Potential Outlier Warning, {name}, {note}({encoding})\n")
                    print(f"Encoding error, {name}, {encoding}, {note}")

    inputs = self._inputs
    outputs = self._outputs
    if isinstance(self._inputs, set):
        inputs = list(self._inputs)
    if isinstance(self._outputs, set):
        outputs = list(self._outputs)
    graph_in_out = inputs + outputs
    for key in graph_in_out:
        if key not in self.encodings:
            csv_info.append(f"Graph Input/Ouput Encoding Missing Warning, {key}, This input or output does not exist in the encoding file.\n")
        else:
            print(f"input/output encoding name is {key}")

    if "lora" in self.model_type:
        lora_count = 0
        for key in self.encodings:
            if "Constant" in key or "lora_alpha" in key:
                const_min = self.encodings[key].min
                const_max = self.encodings[key].max
                csv_info.append(f"Constant Encoding Info, {key}, Potential lora alpha encoding: min={const_min} and max= {const_max}\n")
                lora_count += 1
        if lora_count == 0:
            csv_info.append(f"Encoding missing Error, lora_alpha, Missing LoRA Alpha encoding in the encoding file\n")
        else:
            csv_info.append(f"Constant Encoding Info, , There are {lora_count} constant encodings in total\n")

    reshape_count = 0
    transpose_count = 0
    cast_count = 0
    for name, node in self.nodes.items():
        rules = rules_by_node_exist.get(node.type, None)
        if rules is not None:
            for checker, note in rules:
                try:
                    if checker(node):
                        node.notes.append(note)
                        csv_info.append(f"{node.type} Type Encoding Exist Warning, {node.title},\n")
                        if node.type == 'Reshape':
                            reshape_count += 1
                        elif node.type == 'Transpose':
                            transpose_count += 1
                        elif node.type == 'Cast':
                            cast_count += 1
                        print(f"Encoding error, {node.title}")
                except KeyError as e:
                    node.notes.append(f"KeyError {e}")
                except RuntimeError as e:
                    node.notes.append(f"RuntimeError {e}")
    csv_info.append(f"Reshape Exist Warning Summary, , There are {reshape_count} Reshape encodings\n")
    csv_info.append(f"Transpose Exist Warning Summary, , There are {transpose_count} Transpose encodings\n")
    csv_info.append(f"Cast Exist Warning Summary, , There are {cast_count} Cast encodings\n")


    return csv_info

class OnnxGraph(DotGraph):
    def __init__(self, onnxmodel, encoding, base_dir, show_constant, show_data, model_type):
        super().__init__(encoding)

        self.model_type = model_type
        self.base_dir = base_dir
        self.show_data = show_data
        self._inputs = {i.name for i in onnxmodel.graph.input}
        self._outputs = {i.name for i in onnxmodel.graph.output}

        self.constants.update({param.name: param for param in onnxmodel.graph.initializer})
        self.constants.update({node.output[0]:node for node in onnxmodel.graph.node if node.op_type == 'Constant'})
        if not show_constant:
            # hide 'Identity' opeartors
            self.constants.update({node.output[0]:self.constants[node.input[0]] for node in onnxmodel.graph.node if node.op_type == 'Identity' and node.input[0] in self.constants})

        self.value_infos = {i.name: i for i in onnxmodel.graph.value_info}
        self.value_infos.update({i.name: i for i in onnxmodel.graph.input})
        self.value_infos.update({i.name: i for i in onnxmodel.graph.output})

        nodes = [OnnxNode(node, idx) for idx, node in enumerate(onnxmodel.graph.node) if show_constant or node.output[0] not in self.constants]
        nodes = {node.title: node for node in nodes}
        self._build_graph(nodes)
        if encoding:
            self.csv_info = qnn_encoding_checker(self)

class QnnNode(DotNode):
    def __init__(self, node, node_name, node_index: int):
        super().__init__()
        self.node = node
        self.name = node_name
        self.node_index = node_index
        self.notes = []
        self.type = node['type']
        self._inputs = node["input_names"]
        self._outputs = node["output_names"]
        self._attributes = {}
        self.check_param(node)

    def __getattr__(self, name): # behave like qnn nodes
        return self.node[name]

    def check_param(self, node):
        if 'param_map' in node and 'scalar_params' in node:
            scalar_param_keys = [key for key, val in self.node['param_map'].items() if val == 2 and key != 'packageName']
            for key in scalar_param_keys:
                if key in node['scalar_params']:
                    op_type = next(iter(node['scalar_params'][key].values()))
                    self.type = op_type.replace('ElementWise', '')

        if 'tensor_params' in node:
            tensors = {name:f'{next(iter(tensor.values()))["data"]}' for name, tensor in node['tensor_params'].items()}
            self._attributes.update(tensors)

def _qnn_symmetric_offset(bitwidth):
    return -(2**(bitwidth-1))

class QnnEncoding(Encoding):
    def __init__(self, quant_params):
        super().__init__()
        self._from_quant_params(quant_params)

    def _from_quant_params(self, quant_params):
        if quant_params['definition'] != 1:
            return

        if 'scale_offset' in quant_params:
            self.count = 1
            scale_offset = quant_params['scale_offset']

        elif 'axis_scale_offset' in quant_params:
            self.count = quant_params['axis_scale_offset']['num_scale_offsets']
            scale_offset = quant_params['axis_scale_offset']['scale_offsets'][0]
        else:
            raise ValueError("Couldn't find quantization parameters")

        self.bitwidth = scale_offset['bitwidth']
        self.scale = scale_offset['scale']
        self.offset = scale_offset['offset']
        self.min = scale_offset['minimum']
        self.max = scale_offset['maximum']
        self.symmetric = scale_offset['is_symmetric']

    def match_encoding(self, aimet_encoding, factor=256):

        if aimet_encoding.bitwidth == self.bitwidth and abs(aimet_encoding.scale-self.scale) < 1e-8 and aimet_encoding.offset == self.offset:
            return True, f'{aimet_encoding}'

        delta_s = abs(aimet_encoding.scale-self.scale)
        ratio_s = 100 * delta_s / aimet_encoding.scale
        desc = f'Diff: scale:{_f(delta_s)}[{ratio_s:.1f}%], offset:{abs(aimet_encoding.offset-self.offset)}'
        if aimet_encoding.bitwidth != self.bitwidth:
            if aimet_encoding.bitwidth==8:
                qnn_scale_8b, qnn_offset_8b = self.scale*factor, _round(self.offset/factor)
                delta_s = abs(aimet_encoding.scale-qnn_scale_8b)
                ratio_s = 100 * delta_s / aimet_encoding.scale
                desc = f'=>8bit,(s*{factor},o/{factor}):{_f(qnn_scale_8b)}:{_round(qnn_offset_8b)}'
                desc += f'\nDiff: scale:{_f(delta_s)}[{ratio_s:.1f}%], offset:{abs(aimet_encoding.offset-qnn_offset_8b)}'
                if delta_s < 1e-8 and aimet_encoding.offset==qnn_offset_8b:
                    return True, f'{aimet_encoding}{_br()}{_esc(desc)}'

        return False, f'{aimet_encoding}, mismatch to AIMET'

class QnnGraph(DotGraph):
    # QNN_SDK/include/QNN/QnnTypes.h
    QNN_TENSOR_TYPE_APP_WRITE = 0
    QNN_TENSOR_TYPE_APP_READ = 1
    QNN_TENSOR_TYPE_APP_READ_WRITE = 2
    QNN_TENSOR_TYPE_APP_NATIVE = 3
    QNN_TENSOR_TYPE_APP_STATIC = 4
    QNN_TENSOR_TYPE_APP_NULL = 5
    QNN_TENSOR_TYPE_APP_UPDATEABLE_STATIC = 6
    QNN_TENSOR_TYPE_APP_UPDATEABLE_NATIVE = 7
    QNN_TENSOR_TYPE_APP_UPDATEABLE_APP_WRITE = 8
    QNN_TENSOR_TYPE_APP_UPDATEABLE_APP_READ = 9
    QNN_TENSOR_TYPE_APP_UPDATEABLE_APP_READWRITE = 10

    def __init__(self, qnn_graph, encoding, show_constant, model_type):
        super().__init__(encoding)
        qnntensors, qnnnodes = qnn_graph['graph']['tensors'], qnn_graph['graph']['nodes']

        self.model_type = model_type
        self.tensors = qnntensors
        self._inputs = [name for name, tensor in qnntensors.items() if tensor["type"] == self.QNN_TENSOR_TYPE_APP_WRITE]
        self._outputs = [name for name, tensor in qnntensors.items() if tensor["type"] == self.QNN_TENSOR_TYPE_APP_READ]

        nodes = [QnnNode(node, name, idx) for idx, (name, node) in enumerate(qnnnodes.items())]
        nodes = {node.title: node for node in nodes}
        self._build_graph(nodes)
        self.csv_info = qnn_encoding_checker(self)

    def _find_encoding(self, tensor):
        if tensor not in self.tensors or 'quant_params' not in self.tensors[tensor]:
            return None
        qnn_encoding = QnnEncoding(self.tensors[tensor]['quant_params'])
        return qnn_encoding

class DlcEncoding(Encoding):
    # QnnTypes.h
    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET = 0
    QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET = 1
    QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET = 2
    QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET = 3
    QNN_QUANTIZATION_ENCODING_BLOCK = 4
    QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION = 5
    QNN_QUANTIZATION_ENCODING_VECTOR = 6

    def __init__(self, encoding):
        super().__init__()
        self.is_valid = True
        self._init_from(encoding)

    def _init_from(self, encoding):
        encInfo = None
        self.type = encoding.type.value
        if self.type == DlcEncoding.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
            self.count = 1
            self.desc = ''
            encInfo = encoding.encInfo
        elif self.type == DlcEncoding.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
            info = encoding.axisEncInfo
            self.count = len(info.encInfos)
            self.desc = f"AXIS:{info.axis} "
            encInfo = info.encInfos[0] # show just one
        elif self.type == DlcEncoding.QNN_QUANTIZATION_ENCODING_BLOCK: # BQ
            if hasattr(encoding, 'bqEncInfo'):
                info = encoding.bqEncInfo
                self.count = len(info.encInfos)
                self.desc = f"BQ:ax:{info.axis} blkAx:{info.blockAxis} blkSize:{info.blockSize} " # 'axis', 'blockAxis', 'blockSize', 'encInfos'
                encInfo = info.encInfos[0] # show just one
        elif self.type == DlcEncoding.QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION: # LPBQ
            if hasattr(encoding, 'lpbqEncInfo'):
                info = encoding.lpbqEncInfo
                self.count = len(info.encInfos)
                self.desc = f"LPBQ:ax:{info.axis} blkScBW:{info.blockScaleBitwidth} blk:{info.blockSize}, #blk:{info.numBlocksPerAxis} " # 'axis', 'blockScaleBitwidth', 'blockSize', 'blocksScale16', 'blocksScale8', 'encInfos', 'numBlocksPerAxis'
                encInfo = info.encInfos[0] # show just one
        elif self.type == DlcEncoding.QNN_QUANTIZATION_ENCODING_VECTOR : # VQ
            if hasattr(encoding, 'vqEncInfo'):
                info = encoding.vqEncInfo
                self.count = len(info.encInfos)
                self.desc = f"VQ:ax:{info.axis} rowsPerBlk:{vp.rowsPerBlock}, colsPerBlk:{vp.columnsPerBlock} rowAx:{info.rowAxis} colAx:{info.columnAxis} vD:{info.vectorDimension}, vS:{info.vectorStride}"
                encInfo = info.encInfos[0] # show just one
        elif self.type == DlcEncoding.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:
            info = encoding.axisEncInfo
            self.count = len(info.encInfos)
            self.desc = f"AXIS:{info.axis} "
            encInfo = info.encInfos[0] # show just one
        elif self.type == DlcEncoding.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
            self.count = 1
            self.desc = ''
            encInfo = encoding.encInfo
        else:
            print(f"Encoding info is {encInfo}")
            print(f"Encoding is {encoding}")

        if encInfo is not None:
            self.bitwidth = encInfo.bw
            self.scale = encInfo.scale
            self.offset = encInfo.offset
            self.min = encInfo.min
            self.max = encInfo.max
            self.symmetric = self.offset == _qnn_symmetric_offset(self.bitwidth)
        else:
            self.is_valid = False

    def __repr__(self):
        if not self.is_valid:
            return f'Unknown encoding type:{self.type}'
        return f'{self.desc} q:{_f(self.scale)}:{self.offset}'          \
            f' bw:{self.bitwidth}'                          \
            f' {_f(self.min)}~{_f(self.max)}'               \
            f' {" symmetric" if self.symmetric else ""}'

    def match_encoding(self, aimet_encoding, factor=256):
        if self.type not in (self.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET, self.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET ):
            return None, f'{aimet_encoding}'

        if aimet_encoding.bitwidth == self.bitwidth and abs(aimet_encoding.scale-self.scale) < 1e-8 and aimet_encoding.offset == self.offset:
            return True, f'{aimet_encoding}'
            
        delta_s = abs(aimet_encoding.scale-self.scale)
        ratio_s = 100 * delta_s / aimet_encoding.scale
        desc = f'Diff: scale:{_f(delta_s)}[{ratio_s:.1f}%], offset:{abs(aimet_encoding.offset-self.offset)}'
        if aimet_encoding.bitwidth != self.bitwidth:
            if aimet_encoding.bitwidth==8:
                qnn_scale_8b, qnn_offset_8b = self.scale*factor, _round(self.offset/factor)
                delta_s = abs(aimet_encoding.scale-qnn_scale_8b)
                ratio_s = 100 * delta_s / aimet_encoding.scale
                desc = f'=>8bit,(s*{factor},o/{factor}):{_f(qnn_scale_8b)}:{_round(qnn_offset_8b)}'
                desc += f'\nDiff: scale:{_f(delta_s)}[{ratio_s:.1f}%], offset:{abs(aimet_encoding.offset-qnn_offset_8b)}'
                if delta_s < 1e-8 and aimet_encoding.offset==qnn_offset_8b:
                    return True, f'{_esc(desc)}{_br()}{aimet_encoding}'
        return False, f'{aimet_encoding}, mismatch to AIMET'

class DlcNode(DotNode):
    def __init__(self, node, node_index):
        super().__init__()
        self.node = node
        self.name = node.name
        self.node_index = node_index
        self.notes = []
        self.type = node.type
        self._inputs = [i.name() for i in node.inputs()]
        self._outputs = [i.name() for i in node.outputs()]
        self._attributes = node.attrs_dict


def import_modeltools():
    for env in ['', 'QNN_SDK_ROOT', 'SNPE_ROOT']:
        try:
            if env and os.getenv(env):
                path = f'{os.getenv(env)}/lib/python'
                print(f'Import dlc_utils from {path}')
                sys.path.insert(0, path)
            from qti.aisw.dlc_utils import modeltools
            break
        except NameError as e:
            modeltools = None
            print('NameError', e)
        except ImportError as e:
            modeltools = None
            print('ImportError', e)

    if modeltools is None:
        exit(1)
    return modeltools

class DlcGraph(DotGraph):
    def __init__(self, dlcfile, encoding, show_constant, model_type):
        super().__init__(encoding)
        self.model_type = model_type
        self.model = self._open(dlcfile)
        self._init()
        self.csv_info = qnn_encoding_checker(self)
        

    def _open(self, dlcfile):
        modeltools = import_modeltools()
        model = modeltools.IrDlcReader()
        model.open(dlcfile)
        return model

    def _init(self):
        graph = self.model.get_ir_graph()
        self._inputs = [i.name() for i in graph.get_input_tensors_to_graph()]
        self._outputs = [i.name() for i in graph.get_output_tensors_of_graph()]

        self.tensors = graph.get_tensor_map()

        nodes = [DlcNode(node, idx) for idx, node in enumerate(graph.get_ops())]# if show_constant or node.output[0] not in self.constants]
        nodes = {node.title: node for node in nodes}

        self._build_graph(nodes)

    def _find_encoding(self, tensor):
        dlctensor = self.tensors[tensor]
        encoding = dlctensor.get_encoding()
        if not encoding or encoding.type.name == 'QNN_QUANTIZATION_ENCODING_UNDEFINED':
            return None
        return DlcEncoding(encoding)
    
    def mismatch_dump(self, tensor, csv_info):
        dlc_tensor = tensor
        qnn_encoding = None
        if tensor not in self.producer and tensor not in self.consumers:
            find = False
            for node_name, node in self.nodes.items():
                if tensor == node.name:
                    dlc_tensor = node._outputs[0]
                    find = True
                    break
            if find == False:
                csv_info.append(f"Encoding not found Error, {tensor}, There is no encoding in dlc or net json for this tensor\n")
            else:
                qnn_encoding = self._find_encoding(dlc_tensor)
        else:
            qnn_encoding = self._find_encoding(dlc_tensor)
        if self.encodings and tensor in self.encodings and qnn_encoding is not None:
            match, desc = qnn_encoding.match_encoding(self.encodings[tensor])
            if match == False:
                csv_info.append(f"Encoding mismatch to AIMET Encoding WARNING, {tensor}, QNN Encoding: {qnn_encoding}, AIMET Encoding: {self.encodings[tensor]}\n")
        return csv_info

support_types = ['lora', 'llm-bq', 'llm-lpbq', 'lvm', 'llm']
class ValidateModelType(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        model_types = values.split(',')
        for model_type in model_types:
            if model_type.strip() not in support_types:
                raise argparse.ArgumentError(self, f"Invalid model type: {model_type}. Supported types: {', '.join(support_types)}")
        setattr(namespace, self.dest, model_types)

class ValidateMha2shaMap(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        mha_paths = values.split(',')
        for mha_path in mha_paths:
            if ".json" not in mha_path:
                raise argparse.ArgumentError(self, f"Invalid MHA to SHA Map format: {mha_path}. Need json format")
        setattr(namespace, self.dest, mha_paths)

class ValidateEncodings(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        encodings = values.split(',')
        for encoding in encodings:
            if ".encodings" not in encoding:
                raise argparse.ArgumentError(self, f"Invalid Encoding format: {encoding}. Need json format with .encodings postfix")
        setattr(namespace, self.dest, encodings)

def main(default_output='.csv'):
    parser = argparse.ArgumentParser(description='Encoding Validation')
    parser.add_argument('inputfile', help='onnx file name')
    parser.add_argument('outputfile', nargs='?', help='optional output file name')
    parser.add_argument('-e', '--encodings', default=None, action='append', help='AIMET encoding file name')
    parser.add_argument('-c', '--show-constant', action='store_true', help='Show constants')
    parser.add_argument('-s', '--show-data', action='store_true', help='Show data')
    parser.add_argument('-t', '--model-type', default=[], action=ValidateModelType, help=f'Sepcify the model type, supported types include {support_types}')
    parser.add_argument('-m', '--map-path', default=[], action=ValidateMha2shaMap, help='MHA2SHA encoding map json path')
    parser.add_argument('-2e', '--sec-encodings', default=[], action=ValidateEncodings, help='The second AIMET encoding file name')
    
    args = parser.parse_args()

    print(f"encoding is {args.encodings}")
    print(f"model_type is {args.model_type}")
    print(f"sec_encodings are {args.sec_encodings}")
    print(f"map_paths are {args.map_path}")

    name, ext = os.path.splitext(args.inputfile)
    outputfile = args.outputfile if args.outputfile else f'{name}.csv'
    if not args.encodings:
        if os.path.exists(f'{name}.encodings'):
            args.encodings.append(f'{name}.encodings')
        elif os.path.exists(f'{name}_base.encodings'):
            args.encodings.append(f'{name}_base.encodings')

    csv_info = []
    encoding = load_aimet_encoding(args.encodings, csv_info)
    graph = None

    if args.sec_encodings and not args.map_path:
        sec_encodings = load_aimet_encoding(args.sec_encodings, csv_info)
    elif args.sec_encodings and args.map_path:
        sec_encodings, map_dict = load_sha_aimet_encodings_and_maps(encoding, args.sec_encodings, args.map_path, csv_info)

    if ext == '.onnx':
        base_dir = os.path.dirname(args.inputfile)
        onnxmodel = OnnxModel(args.inputfile)
        graph = OnnxGraph(onnxmodel, encoding, base_dir, show_constant=args.show_constant, show_data=args.show_data, model_type=args.model_type)
    elif ext == '.json':
        with open(args.inputfile, 'rt') as f:
            qnn_graph = json.load(f)
        graph = QnnGraph(qnn_graph, encoding, show_constant=args.show_constant, model_type=args.model_type)
    elif ext == '.dlc':
        graph = DlcGraph(args.inputfile, encoding, show_constant=args.show_constant, model_type=args.model_type)
    
    if graph is not None:
        if args.sec_encodings and not args.map_path:
            csv_info += encoding_validation(encoding, sec_encodings, graph.csv_info)
        elif args.sec_encodings and args.map_path:
            csv_info += validate_mha_sha_encodings(encoding, sec_encodings, map_dict, graph.csv_info)
        else:
            csv_info += graph.csv_info

    with open(outputfile, 'wt') as f:
        print(f'Save {outputfile}')
        for item in csv_info:
            f.write(item)

if __name__ == '__main__':
    main()