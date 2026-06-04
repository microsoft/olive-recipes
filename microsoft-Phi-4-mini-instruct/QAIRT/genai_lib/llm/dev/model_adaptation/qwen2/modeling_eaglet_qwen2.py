#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from torch import nn
from genai_lib.llm.dev.model_adaptation.qwen2.adaptation import adapted_update_causal_mask, QcQwen2Attention
from genai_lib.llm.eaglet.base_draft_model import BaseDraftModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer

__all__ = ["Qwen2EagletDraftModel", "Qwen2EagletDecoderLayer"]


class Qwen2EagletDraftModel(BaseDraftModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(
            config,
            decoder_cls=Qwen2EagletDecoderLayer,
            norm_cls=Qwen2RMSNorm,
            dual_fc=True,
        )

    _update_causal_mask = adapted_update_causal_mask


class Qwen2EagletDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        if layer_idx == 0:
            self.input_layernorm = nn.Identity()
        self.self_attn = QcQwen2Attention(config=config, layer_idx=layer_idx)