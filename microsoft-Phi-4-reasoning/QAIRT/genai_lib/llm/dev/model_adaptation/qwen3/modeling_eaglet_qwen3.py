#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" 
Eaglet model for Qwen3
"""

from torch import nn
from genai_lib.llm.dev.model_adaptation.qwen3.adaptation import QcQwen3Attention
from genai_lib.llm.eaglet.base_draft_model import Eaglet2BaseDraftModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm


__all__ = ["Qwen3Eaglet2DraftModel", "Qwen3EagletDecoderLayer"]

class Qwen3Eaglet2DraftModel(Eaglet2BaseDraftModel):
    def __init__(self, config):
        super().__init__(config, decoder_cls=Qwen3EagletDecoderLayer, norm_cls=Qwen3RMSNorm)


class Qwen3EagletDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        if layer_idx == 0:
            self.input_layernorm = nn.Identity()
        self.self_attn = QcQwen3Attention(config=config, layer_idx=layer_idx)