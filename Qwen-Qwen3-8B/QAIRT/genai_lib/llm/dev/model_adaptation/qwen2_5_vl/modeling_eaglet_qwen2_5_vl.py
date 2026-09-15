#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""
Eaglet model for Qwen2_5_VL
"""

from torch import nn
from genai_lib.llm.dev.model_adaptation.qwen2_5_vl.adaptation import QcQwen2_5_VLAttention
from genai_lib.llm.eaglet.v1.base_draft_model import BaseDraftModel
from genai_lib.llm.eaglet.v2.base_draft_model import BaseDraftModel as Eaglet2BaseDraftModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2RMSNorm


__all__ = ["Qwen2_5_VLEagletDraftModel", "Qwen2_5_VLEaglet2DraftModel", "Qwen2_5_VLEagletDecoderLayer"]


class Qwen2_5_VLEagletDraftModel(BaseDraftModel):
    def __init__(self, config):
        super().__init__(
            config, decoder_cls=Qwen2_5_VLEagletDecoderLayer, norm_cls=Qwen2RMSNorm, dual_fc=True
        )


class Qwen2_5_VLEaglet2DraftModel(Eaglet2BaseDraftModel):
    def __init__(self, config):
        super().__init__(config, decoder_cls=Qwen2_5_VLEagletDecoderLayer, norm_cls=Qwen2RMSNorm)


class Qwen2_5_VLEagletDecoderLayer(Qwen2_5_VLDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        if layer_idx == 0:
            self.input_layernorm = nn.Identity()
        self.self_attn = QcQwen2_5_VLAttention(config=config, layer_idx=layer_idx)
