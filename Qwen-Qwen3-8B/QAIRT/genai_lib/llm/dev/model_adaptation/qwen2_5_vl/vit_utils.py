#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


'''
Utils required for Qwen 2.5VL ViT Onboarding
'''


import torch
import functools
import torch.nn.functional as F
from genai_lib.common.dev.utils import rsetattr



def vit_prepare_attention_mask(inputs_tensor, cu_seqlens, mask_neg = -1e3) -> torch.Tensor:
    '''
    We yank out the attention mask preparation for ViT attention from transformers 4.53.1
    Refer to `Qwen2_5_VisionTransformerPretrainedModel._prepare_attention_mask`
    Transformers recent version apply attention on each window separately which excessively tiles the computation graph
    Hence we stick to the transformers<=4.53 way of creating the windowed-attention aware attention mask
    '''

    seq_length = inputs_tensor.shape[0]
    attention_mask = torch.full(
        [1, 1, seq_length, seq_length],
        torch.finfo(inputs_tensor.dtype).min,
        device=inputs_tensor.device,
        dtype=inputs_tensor.dtype,
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    attention_mask = attention_mask.clamp_min(mask_neg)

    return attention_mask



def vit_get_position_embeddings(visual_model, hidden_states, grid_thw):
    '''
    Precompute RoPE position embeddings
    We yank this section out from `Qwen2_5_VisionTransformerPretrainedModel.forward`
    '''

    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
    position_embeddings_cos = rotary_pos_emb.cos()
    position_embeddings_sin = rotary_pos_emb.sin()

    return position_embeddings_cos.to(hidden_states.device), position_embeddings_sin.to(hidden_states.device)



class Conv2dsInplaceConv3d(torch.nn.Module):
    """
    Adaptation layer that replaces a 3D convolution (`torch.nn.Conv3d`) with multiple 2D convolutions.

    This is useful when you want to decompose a 3D convolution along one spatial dimension (time, height, or width)
    and apply separate 2D convolutions to each slice, then aggregate the results.

    Note that this adaptation was specifically designed for Qwen VLMs and might need changes if being referred to for other models.

    Args:
        module (torch.nn.Conv3d): The original Conv3d module to adapt.
        split_dim (int): The spatial kernel dimension to split along (0 = time/depth T, 1 = height H, 2 = width W).

    Attributes:
        bias (torch.Tensor or None): Bias from the original Conv3d, if present.
        split_dim (int): Dimension along which the kernel and input are split.
        weight (torch.Tensor): Placeholder tensor for compatibility with code that expects `.weight`.
        convs (torch.nn.ModuleList): List of Conv2d layers corresponding to each slice of the 3D kernel.

    Raises:
        TypeError: If `module` is not an instance of `torch.nn.Conv3d`.
        ValueError: If `split_dim` is not one of {0, 1, 2}.
    """

    def __init__(self, module: torch.nn.Conv3d, split_dim: int = 0):

        # Checking assumptions behind the adaptation
        if not isinstance(module, torch.nn.Conv3d):
            raise TypeError(
                f"Conv2dsInplaceConv3d can only replace torch.nn.Conv3d modules. Got {type(module).__name__} instead."
            )
        if split_dim not in (0, 1, 2):
            raise ValueError(
                f"Invalid split_dim {split_dim}: must be 0 (time), 1 (height), or 2 (width) for 3D convolution kernels."
            )

        super().__init__()

        # Copy basic attributes from original Conv3d
        self.bias = module.bias  # Keep original bias (or None)
        self.split_dim = split_dim

        # Compute 2D kernel size by removing the split dimension
        kernel_size = (
            module.kernel_size[:self.split_dim] +
            module.kernel_size[self.split_dim + 1 :]
        )

        # Placeholder for `.weight` to satisfy external code (e.g., HuggingFace models)
        # Ensures correct dtype but zero elements
        self.weight = torch.empty(0, dtype=module.weight.dtype)

        # Arguments for each Conv2d slice
        args = {
            "out_channels": module.out_channels,
            "in_channels": module.in_channels,
            "kernel_size": kernel_size,  # Sliced kernel_size with split_dim removed
            "stride": kernel_size,  # The original model implementation keeps stride equal to kernel size, which holds true for split kernels as well
            "bias": False,  # Bias handled separately
        }

        # Create a ModuleList of Conv2d layers
        self.convs = torch.nn.ModuleList()
        for i in range(module.kernel_size[self.split_dim]):
            conv = torch.nn.Conv2d(**args)
            # Copy corresponding slice of weights from Conv3d
            conv.weight.data.copy_(
                torch.narrow(module.weight.data, -3 + self.split_dim, i, 1)
                .squeeze(-3 + self.split_dim)
            )
            self.convs.append(conv)

        # Move all submodules and buffers to the same device as original weights
        self.to(module.weight.data.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: split the input along the chosen spatial dimension, apply each Conv2d slice,
        sum the outputs, and add bias if present.

        **Qwen 2.5VL-specific assumption (Vision Patch Embed setting):**
            - The adapter expects the input spatial-temporal size (T, H, W) to be exactly equal to the
              effective 3D kernel size implied by the adapter. If this equality does not hold, the forward pass raises an error.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, T, H, W).

        Returns:
            torch.Tensor: Output tensor after applying all 2D convolutions and summing.

        Raises:
            ValueError: If the input (T, H, W) does not exactly match the effective 3D kernel size.
                        This adapter is specific to the Qwen 2.5VL Vision Patch Embed configuration.
        """

        kernel_3d_size = torch.Size((
            *self.convs[0].kernel_size[: self.split_dim],
            len(self.convs),
            *self.convs[0].kernel_size[self.split_dim :]
        ))  # Insert number of 2D convs at split_dim index
        input_thw = x.shape[-3:]
        error_msg = (
            f"This adaptation is specific to Qwen 2.5VL style Vision Patch Embed setting "
            f"where the size of the Conv3d kernel must equal the (T, H, W) values of the input. "
            f"Received input THW {input_thw}, expected kernel size {kernel_3d_size}."
        )
        if input_thw != kernel_3d_size:
            raise ValueError(error_msg)

        # Split input into slices along the chosen dimension
        slices = torch.split(x, 1, dim = -3 + self.split_dim)

        outputs = []
        for i, conv in enumerate(self.convs):
            # Remove the split dimension (size=1) for Conv2d input
            cur_slice = slices[i].squeeze(-3 + self.split_dim)
            out = conv(cur_slice)

            # Add bias if present (reshape for broadcasting)
            # We add bias to the first conv since bias is not quantized and conv is typically implemented as a single kernel
            if i == 0 and self.bias is not None:
                out += self.bias.view(1, -1, 1, 1)
            outputs.append(out)

        # Aggregate outputs by summation
        result = sum(outputs)

        return result



def replace_conv3d_with_conv2ds(model: torch.nn.Module, split_dim: int = 0):
    """
    A helper function that replaces `Conv3d` modules with `Conv2dsInplaceConv3d`

    Args:
        model (torch.nn.Module): Model in which Conv3d need to be replaced.
        split_dim (int): Dimension along which the kernel and input are split. Belongs to [0, 1, 2] since we are dealing with 3D Convolution Kernels.
    """

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            conv_layer = Conv2dsInplaceConv3d(module, split_dim)
            rsetattr(model, name, conv_layer)



@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    config = AutoConfig.from_pretrained(model_id_or_path)
    config.num_hidden_layers = config.text_config.num_hidden_layers = config.vision_config.depth = 1
    config.vision_config.fixed_grid_thw = [1, 1, 1]  # Set a dummy fixed_grid_thw since the adapted ViT class `QcQwen2_5_VisionTransformerPretrainedModel` expects this attribute in initialization
    model = Qwen2_5_VLModel(config)
    return model
