#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

# =============================================================================
# coding=utf-8
# Copyright 2024 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from torch import nn
from genai_lib.llm.dev.model_adaptation.glm4v.adaptation import adapted_update_causal_mask, QcGlmAttention
from genai_lib.llm.eaglet.base_draft_model import BaseDraftModel

try:
    from transformers_modules import modeling_glm
except ImportError as e:
    print(f"{e} \n Please instantiate Glm4v with AutoModelForCausalLM first to have transformers_modules cache")

__all__ = ["GlmEagletDraftModel", "GlmEagletDecoderLayer"]


class GlmEagletDraftModel(BaseDraftModel):
    def __init__(self, config):
        super().__init__(
            config,
            decoder_cls=GlmEagletDecoderLayer,
        )

    _update_causal_mask = adapted_update_causal_mask


class GlmEagletDecoderLayer(modeling_glm.GlmDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        if layer_idx == 0:
            self.input_layernorm = nn.Identity()
        self.self_attn = QcGlmAttention(config=config, layer_idx=layer_idx)