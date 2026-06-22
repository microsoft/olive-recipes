#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch
from diffusers.models.autoencoders.vae import DecoderOutput

class VAEDecoderMerged(torch.nn.Module):
    '''
    Unify decoding steps from vae.post_quant_conv and vae.decoder
    into a single ONNX graph
    '''
    def __init__(self, post_quant_conv, decoder):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder
        dtype = next(iter(decoder.parameters())).dtype
        self.to(dtype)

    def __getattr__(self, attr):
        if attr in set(self._modules.keys()):
            return self._modules[attr]
        return getattr(self._modules['decoder'], attr)

    def forward(self, z: torch.FloatTensor, return_dict: bool = True):
        '''
        Adapted from https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/models/autoencoders/autoencoder_kl.py#L270-L280
        '''
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)