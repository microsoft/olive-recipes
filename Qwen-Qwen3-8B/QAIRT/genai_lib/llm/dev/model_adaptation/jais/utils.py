#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch
from genai_lib.llm.dev.model_adaptation.gpt2.utils import llm_update_causal_mask


def llm_create_position_embeddings(position_id, past_length):
    position_id = position_id - position_id.min() + past_length
    context_position = position_id.view(-1, 1)
    memory_position = torch.concat([torch.arange(past_length, device=position_id.device).view(1, -1), position_id], dim=1)
    relative_position = memory_position - context_position
    relative_position = torch.abs(relative_position)
    return relative_position
