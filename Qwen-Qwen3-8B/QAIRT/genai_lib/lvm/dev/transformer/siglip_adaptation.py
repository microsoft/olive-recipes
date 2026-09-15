# /usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" model adaptations for optimal accuracy and latency when quantizing and executing on constrained hardware """

import functools
import math
import torch
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionEmbeddings
from aimet_torch.nn.modules import custom as elementwise_ops
from peft.tuners.lora.layer import Linear as LoraLinear
from aimet_torch.peft import LoraLayer as AimetLoraLayer


@torch.no_grad
def _convert_linear_to_conv(linear, scale=None):
    conv = torch.nn.Conv2d(linear.weight.shape[1], linear.weight.shape[0], 1, bias=linear.bias is not None)
    conv.weight.copy_(linear.weight.data[..., None, None] * scale if scale is not None else
                      linear.weight.data[..., None, None])
    if linear.bias is not None:
        conv.bias.copy_(linear.bias.data * scale if scale is not None else linear.bias.data)
    return conv


class SelfAttention4d(torch.nn.Module):

    def __init__(self,
                 self_attn):
        super(SelfAttention4d, self).__init__()
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.out_proj = _convert_linear_to_conv(self_attn.out_proj)
        self.scale = self_attn.scale

        self.num_heads = self_attn.num_heads
        self.head_dim = self_attn.head_dim

        if isinstance(self.q_proj, ((AimetLoraLayer, LoraLinear))):
            self.q_proj = self._replace_lora_linear_with_conv(self.q_proj)
            self.k_proj = self._replace_lora_linear_with_conv(self.k_proj)
            self.v_proj = self._replace_lora_linear_with_conv(self.v_proj)
        else:
            q_proj, k_proj, v_proj = self.q_proj, self.k_proj, self.v_proj
            self.q_proj = _convert_linear_to_conv(q_proj)
            self.k_proj = _convert_linear_to_conv(k_proj)
            self.v_proj = _convert_linear_to_conv(v_proj)

            del q_proj
            del k_proj
            del v_proj

        del self_attn.out_proj

    def forward(self, hidden_states, attn_mask=None):
        batch_size, channel, height, width = hidden_states.shape
        _query = self.q_proj(hidden_states).permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) * self.scale # -> (B, num_heads, H*W, head_dim)

        _key_T = self.k_proj(hidden_states).reshape(batch_size, self.num_heads, self.head_dim, -1) # -> (B, num_heads, head_dim, H*W)

        _value = self.v_proj(hidden_states).reshape(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # -> (B, num_heads, head_dim, H*W) -> (B, num_heads, H*W, head_dim)

        _attn = elementwise_ops.MatMul()(_query, _key_T)  # -> (B, num_heads, H*W, H*W)
        _attn = torch.nn.functional.softmax(_attn, dim=-1)
        _out = elementwise_ops.MatMul()(_attn, _value) # -> (B, num_heads, H*W, head_dim)

        # These permute reshape will be cleaned up in MHA2SHA-onnx-converter
        _out = _out.permute(0, 2, 1, 3) # -> (B, H*W, head, d)
        hidden_states = _out.reshape(batch_size, height, width, -1) # (B, H*W, head, d) -> (B, H, W, C)

        # linear proj
        hidden_states = hidden_states.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        hidden_states = self.out_proj(hidden_states) # -> (B, C, H, W)

        return hidden_states

    @torch.no_grad
    def _replace_lora_linear_with_conv(self, lora_layer, scale=None):
        def _replace_lora_A_or_B(lora_A_or_B, scale=None):
            if isinstance(lora_A_or_B, torch.nn.ModuleList):
                lora_A_or_B_conv = torch.nn.ModuleList()
                for mod in lora_A_or_B:
                    mod_conv = _convert_linear_to_conv(mod, scale)
                    lora_A_or_B_conv.append(mod_conv)
            elif isinstance(lora_A_or_B, torch.nn.ModuleDict):
                lora_A_or_B_conv = torch.nn.ModuleDict()
                for name, mod in lora_A_or_B.items():
                    mod_conv = _convert_linear_to_conv(mod, scale)
                    lora_A_or_B_conv[name] = mod_conv
            else:
                raise ValueError(f"{lora_A_or_B} should be LoRA Layer, got: {type(lora_A_or_B)}")
            return lora_A_or_B_conv

        base_layer, lora_A, lora_B = lora_layer.base_layer, lora_layer.lora_A, lora_layer.lora_B
        base_layer_conv = _convert_linear_to_conv(base_layer, scale)

        lora_layer.base_layer = base_layer_conv
        lora_layer.lora_A = _replace_lora_A_or_B(lora_A, scale=scale)
        lora_layer.lora_B = _replace_lora_A_or_B(lora_B, scale=None)

        if hasattr(lora_layer, "scaling"):
            # Expand the lora alpha vector to be 4d
            for i, scaling in enumerate(lora_layer.scaling):
                if isinstance(scaling, torch.Tensor) and scaling.size():
                    lora_layer.scaling[i].data = scaling.data.view(1, scaling.size()[0], 1, 1)

        del base_layer
        del lora_A
        del lora_B

        return lora_layer


class Mlp4d(torch.nn.Module):

    def __init__(self,
                 siglip_mlp):
        super(Mlp4d, self).__init__()
        self.fc1 = _convert_linear_to_conv(siglip_mlp.fc1)
        self.activation_fn = siglip_mlp.activation_fn
        self.fc2 = _convert_linear_to_conv(siglip_mlp.fc2)
        del siglip_mlp.fc1
        del siglip_mlp.fc2

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerBlock4d(torch.nn.Module):

    def __init__(self,
                 siglip_encoder_layer):
        super(TransformerBlock4d, self).__init__()
        self.layer_norm1 = LayerNorm4d(siglip_encoder_layer.layer_norm1)
        self.self_attn = SelfAttention4d(siglip_encoder_layer.self_attn)
        self.layer_norm2 = LayerNorm4d(siglip_encoder_layer.layer_norm2)
        self.mlp = Mlp4d(siglip_encoder_layer.mlp)

    def forward(self, hidden_states, attention_mask=None, output_attentions=None, **kwargs):
        if not hidden_states.ndim == 4:
            raise NotImplementedError(f'{self.__class__.__name__} expects 4d channels first inputs, but got {hidden_states.ndim}-dimensional inputs')

        output_attentions_kwarg = kwargs.get('output_attentions', output_attentions)
        if not (output_attentions_kwarg is None or output_attentions_kwarg is False):
            raise NotImplementedError(f'{self.__class__.__name__} has not implemented support for output_attentions=True option')

        for k, v in kwargs.items():
            if not (v is None or v is False):
                raise NotImplementedError(f'{self.__class__.__name__} has not implemented support param {k} with value {v}.')

        attn_out = self.layer_norm1(hidden_states)
        attn_out = self.self_attn(attn_out)
        attn_out += hidden_states
        mlp_out = self.layer_norm2(attn_out)
        mlp_out = self.mlp(mlp_out)
        mlp_out += attn_out
        return mlp_out, None


class SiglipVisionEmbeddings4d(torch.nn.Module):
    """ Adapted from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/models/siglip/modeling_siglip.py#L250-L318 """
    def __init__(self, siglip_vision_embeddings):
        super().__init__()
        self.siglip_vision_embeddings = siglip_vision_embeddings

    def forward(self, x, **kwargs):
        for k, v in kwargs.items():
            if not (v is None or v is False):
                raise NotImplementedError(f'{self.__class__.__name__} cannot support param {k} with value {v}.')

        x = self.siglip_vision_embeddings(x)
        B, emb_dim, C = x.shape
        emb_dim_sqrt = int(math.sqrt(emb_dim)) # 4D handling from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/models/siglip/modeling_siglip.py#L295-L296
        x = x.permute(0,2,1)
        x = x.reshape(B, C, emb_dim_sqrt, emb_dim_sqrt)
        return x


class LayerNorm4d(torch.nn.Module):
    def __init__(self, layer_norm):
        super().__init__()
        self.layer_norm = layer_norm

    def forward(self, x, **kwargs):
        for k, v in kwargs.items():
            if not (v is None or v is False):
                raise NotImplementedError(f'{self.__class__.__name__} has not implemented support param {k} with value {v}.')

        return self.layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class GLU4d(torch.nn.Module):
    def __init__(self, GLU):
        super().__init__()
        self.linear_proj = _convert_linear_to_conv(GLU.linear_proj)
        self.norm1 = LayerNorm4d(GLU.norm1)
        self.act1 = GLU.act1
        self.act2 = GLU.act2
        self.dense_h_to_4h = _convert_linear_to_conv(GLU.dense_h_to_4h)
        self.gate_proj = _convert_linear_to_conv(GLU.gate_proj)
        self.dense_4h_to_h = _convert_linear_to_conv(GLU.dense_4h_to_h)

        del GLU.linear_proj
        del GLU.dense_h_to_4h
        del GLU.gate_proj
        del GLU.dense_4h_to_h

    def forward(self, x, **kwargs):
        for k, v in kwargs.items():
            if not (v is None or v is False):
                raise NotImplementedError(f'{self.__class__.__name__} has not implemented support param {k} with value {v}.')

        B, C, H, W = x.shape
        if not H == W:
            raise NotImplementedError(f'{self.__class__.__name__} only supports square 4d inputs, but got height = {H}, and width = {W}.')

        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)

        return x


class Adapter4d(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.conv = adapter.conv
        self.boi = adapter.boi
        self.eoi = adapter.eoi
        self.linear_proj = GLU4d(adapter.linear_proj)

    def forward(self, image_emb, **kwargs):
        for k, v in kwargs.items():
            if not (v is None or v is False):
                raise NotImplementedError(f'{self.__class__.__name__} has not implemented support param {k} with value {v}.')

        B, C, H, W = image_emb.shape
        if not H == W:
            raise NotImplementedError(f'{self.__class__.__name__} only supports square 4d inputs, but got height = {H}, and width = {W}.')

        image_emb = self.conv(image_emb)
        image_emb = self.linear_proj(image_emb)
        image_emb = image_emb.flatten(2).transpose(1, 2)
        image_emb = torch.cat([self.boi.repeat(len(image_emb), 1, 1), image_emb, self.eoi.repeat(len(image_emb), 1, 1)], dim=1)
        return image_emb


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def align_siglip_vision_model_tensor_dimensionlity(model):
    for name, module in model.named_modules():

        if isinstance(module, SiglipVisionEmbeddings):
            embedding_4d = SiglipVisionEmbeddings4d(module)
            rsetattr(model, name, embedding_4d)

        if isinstance(module, SiglipEncoderLayer):
            transfomer_block_4d = TransformerBlock4d(module)
            rsetattr(model, name, transfomer_block_4d)

        if 'post_layernorm' in name:
            layer_norm_4d = LayerNorm4d(module)
            rsetattr(model, name, layer_norm_4d)
