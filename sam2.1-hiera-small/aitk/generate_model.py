# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

import onnx
import requests
import torch
import torch.nn.functional as F
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from onnxsim import simplify
from sam2.build_sam import build_sam2
from sam2.modeling.backbones.hieradet import do_pool
from torch import nn


def download_file(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)


class Conv2DInplaceLinear(nn.Module):
    """An implementation of Linear / Conv1D that uses a 1x1 Conv2D op instead.

    The Conv2D implementation for Qualcomm DSPs is faster than the Linear/Conv1D implementation.
    """

    @staticmethod
    def from_linear(mod: torch.nn.Linear | torch.nn.Conv1d):
        weight: torch.Tensor | torch.nn.Parameter
        bias: torch.Tensor | torch.nn.Parameter | None
        if isinstance(mod, torch.nn.Linear):
            weight, bias = mod.weight, mod.bias
            bias = mod.bias
        elif isinstance(mod, torch.nn.Conv1d):
            weight, bias = mod.weight.T, mod.bias
        else:
            raise NotImplementedError

        out_features, in_features = weight.shape
        linear = Conv2DInplaceLinear(
            in_features,
            out_features,
            bias is not None,
            mod.device if hasattr(mod, "device") else None,
        )
        linear.conv2d.weight.data.copy_(weight.data[:, :, None, None])
        if bias is not None:
            assert linear.conv2d.bias is not None
            linear.conv2d.bias.data.copy_(bias.data)

        return linear

    def __init__(
        self,
        in_features,
        out_features,
        has_bias: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_features, out_features, 1, bias=has_bias)
        if device:
            self.conv2d.to(device)

    def __getattr__(self, attr):
        conv2d = self._modules["conv2d"]
        if attr == "conv2d":
            return conv2d
        return getattr(conv2d, attr)

    def forward(self, x: torch.Tensor):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(1)
        elif ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            pass

        x = x.permute(0, 3, 1, 2)  # (B, L, D) -> (B, D, 1, L)
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1)

        if ndim == 2:
            x = x.squeeze(1).squeeze(0)
        elif ndim == 3:
            x = x.squeeze(1)
        elif ndim == 4:
            pass
        return x


class SplitHeadSAMEncoderAttention(nn.Module):
    """SAM Attention block with the following modifications necessary to run on QNN.

    * Heads are split into separate ops, rather than all heads running in a single op.
    * QKV is unpacked from 1 tensor into 3 tensors.
    """

    def __init__(self, attention_block) -> None:
        super().__init__()
        self.out_feature, self.in_feature = (
            attention_block.qkv.weight.shape[0] // 3,
            attention_block.qkv.weight.shape[1],
        )

        bias = attention_block.qkv.bias[: self.out_feature] is not None
        self.q = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.k = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.v = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.proj = Conv2DInplaceLinear.from_linear(attention_block.proj)
        self.num_heads = attention_block.num_heads
        self.q_pool = attention_block.q_pool

        for chunk, proj_list in enumerate([self.q, self.k, self.v]):
            proj_list.conv2d.weight.data.copy_(
                attention_block.qkv.weight[
                    (chunk) * self.out_feature : (chunk + 1) * self.out_feature,
                    :,
                    None,
                    None,
                ]
            )

            assert proj_list.conv2d.bias is not None
            proj_list.conv2d.bias.data.copy_(
                attention_block.qkv.bias[(chunk) * self.out_feature : (chunk + 1) * self.out_feature,]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        """
        #original code
        # qkv with shape (b, h * w, 3, nHead, C)
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1)
        # q, k, v with shape (b, h * w, nheads, C)
        q, k, v = torch.unbind(qkv, 2)
        """
        k = self.k(x).reshape(b, h * w, self.num_heads, -1).permute(0, 2, 1, 3).reshape(b * self.num_heads, h * w, -1)
        v = self.v(x).reshape(b, h * w, self.num_heads, -1).permute(0, 2, 1, 3).reshape(b * self.num_heads, h * w, -1)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = self.q(x)
            q = do_pool(q, self.q_pool)
            h, w = q.shape[1:3]  # downsampled shape
            q = q.reshape(b, h * w, self.num_heads, -1)
            q = q.permute(0, 2, 1, 3)
            q = q.reshape(b * self.num_heads, h * w, -1)
        else:
            q = (
                self.q(x)
                .reshape(b, h * w, self.num_heads, -1)
                .permute(0, 2, 1, 3)
                .reshape(b * self.num_heads, h * w, -1)
            )
        # Torch's SDPA expects [b, nheads, h*w, C] so we transpose
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )

        # Transpose back
        x = x.reshape(b, self.num_heads, h * w, -1)
        x = x.transpose(1, 2)
        x = x.reshape(b, h, w, -1)

        return self.proj(x)


class Conv2DInplaceLinearSAMTransformerMLPBlock(nn.Module):
    """SAM MLPBlock that uses 1x1 Conv2D in place of linear layers."""

    def __init__(self, mlp_block) -> None:
        super().__init__()
        self.lin1 = Conv2DInplaceLinear.from_linear(mlp_block.layers[0])
        self.lin2 = Conv2DInplaceLinear.from_linear(mlp_block.layers[1])
        self.act = mlp_block.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


def window_partition(x, window_size):
    """Partition into non-overlapping windows with padding if needed.

    Args:
        x (tensor): input tokens with [b, h, w, c].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [b * num_windows, window_size, window_size, c].
        (height_p, width_p): padded h and w before partition

    """
    b, h, w, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    height_p, width_p = h + pad_h, w + pad_w

    x = x.view(height_p // window_size, window_size, width_p // window_size, window_size, c)
    windows = x.permute(0, 2, 1, 3, 4).reshape(-1, window_size, window_size, c)
    return windows, (height_p, width_p)


def window_unpartition(x, window_size, pad_hw, hw):
    """Window unpartition into original sequences and removing padding.

    Args:
        x (tensor): input tokens with [b * num_windows, window_size, window_size, c].
        window_size (int): window size.
        pad_hw (Tuple): padded h and w (height_p, width_p).
        hw (Tuple): original h and w (h, w) before padding.

    Returns:
        x: unpartitioned sequences with [b, h, w, c].

    """
    height_p, width_p = pad_hw
    h, w = hw
    b = x.shape[0] // (height_p * width_p // window_size // window_size)
    x = x.reshape(height_p // window_size, width_p // window_size, window_size, window_size, -1)
    x = x.permute(0, 2, 1, 3, 4).reshape(b, height_p, width_p, -1)

    if height_p > h or width_p > w:
        x = x[:, :h, :w, :]
    return x


class ModMultiScaleBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.model = block
        self.model.mlp = Conv2DInplaceLinearSAMTransformerMLPBlock(self.model.mlp)
        self.model.attn = SplitHeadSAMEncoderAttention(self.model.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, h, w, C
        x = self.model.norm1(x)

        # Skip connection
        if self.model.dim != self.model.dim_out:
            shortcut = do_pool(self.model.proj(x), self.model.pool)

        # Window partition
        window_size = self.model.window_size
        if window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.model.attn(x)
        if self.model.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.model.window_size // self.model.q_stride[0]
            h, w = shortcut.shape[1:3]

            pad_h = (window_size - h % window_size) % window_size
            pad_w = (window_size - w % window_size) % window_size
            pad_hw = (h + pad_h, w + pad_w)

        # Reverse window partition
        if self.model.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (h, w))

        x = shortcut + self.model.drop_path(x)
        # MLP
        return x + self.model.drop_path(self.model.mlp(self.model.norm2(x)))


class SAM2Encoder(nn.Module):
    """Exportable SAM2 encoder that can be split into several parts."""

    def __init__(
        self,
        sam2,
    ) -> None:
        super().__init__()
        self.model = sam2
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        for i, block in enumerate(self.model.image_encoder.trunk.blocks):
            self.model.image_encoder.trunk.blocks[i] = ModMultiScaleBlock(block)

    def forward(self, Image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run SAM2 Image encoder and returns image_embeddings, high_res_features1, high_res_features2.

        Args:
            Image:
                Raw floating point pixel values for encoder consumption.
                3-channel Color Space: RGB, range [0, 1]

        Returns:
                image_embeddings: Shape (1, 256, 64, 64)
                high_res_features1: Shape (1, 32, 256, 256)
                high_res_features2: Shape (1, 64, 128, 128)

        """
        # x = self.normalize(Image)
        x = Image
        backbone_out = self.model.forward_image(x)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        image_embeddings = feats[2]
        high_res_features1 = feats[0]
        high_res_features2 = feats[1]
        return image_embeddings, high_res_features1, high_res_features2


class SAM2Decoder(nn.Module):
    """SAM2Decoder is taken from the class SAM2ImagePredictor.predict from sam2.

    This removes output mask resizing. Because this requires a dynamic shape to accomplish
    in the network, it's better to do this as a postprocessing step rather than in the inference
    framework itself.
    """

    def __init__(self, sam2) -> None:
        super().__init__()
        self.model = sam2
        self.mask_decoder = self.model.sam_mask_decoder
        self.prompt_encoder = self.model.sam_prompt_encoder

    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1,256,64,64]
        high_res_features1: torch.Tensor,  # [1, 32, 256, 256]
        high_res_features2: torch.Tensor,  # [1, 64, 128, 128]
        unnorm_coords: torch.Tensor,  # [num_labels,num_points,2]
        labels: torch.Tensor,  # [num_labels,num_points]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run SAM2 lightweight decoder and return generated mask for given points.

        Args:
            image_embeddings: torch.Tensor of shape [1, emb_dim, emb_size, emb_size]
                Image embeddings generated by Encoder
            high_res_features1: torch.Tensor of shape [1, high_res_1_dim, high_res_1_size, high_res_1_size]
                First set of high-resolution features.
            high_res_features2: torch.Tensor of shape [1, high_res_2_dim, high_res_2_size, high_res_2_size]
                Second set of high-resolution features.
            unnorm_coords: torch.Tensor of shape [1, k, 2]
                Point coordinates from input image for segmentation, mapped to the resized image
            labels: torch.Tensor of shape [1, k]
                Point Labels to select/de-select given point for segmentation
                e.g. Corresponding value is 1 if this point is to be included, otherwise 0

        Returns:
            masks: torch.Tensor of shape [1, 1, 256, 256]
            scores: torch.Tensor of shape [1, 1]

        """
        sparse_embedding, dense_embedding = self.prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=[high_res_features1, high_res_features2],
        )
        return low_res_masks, iou_predictions


model_weights_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
model_config_url = "https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_s.yaml"
checkpoint = "sam2.1_hiera_small.pt"
model_cfg = "sam2.1_hiera_s.yaml"


def main():
    download_file(model_weights_url, checkpoint)
    download_file(model_config_url, model_cfg)

    GlobalHydra.instance().clear()
    initialize(config_path="./", job_name="sam2_inference", version_base=None)

    sam2_model = build_sam2(model_cfg, checkpoint, device="cpu")
    encoder = SAM2Encoder(sam2_model)
    decoder = SAM2Decoder(sam2_model)
    en_inputs = {"Image": torch.rand((1, 3, 1024, 1024))}
    de_inputs = {
        "image_embeddings": torch.rand([1, 256, 64, 64]),
        "high_res_features1": torch.rand([1, 32, 256, 256]),
        "high_res_features2": torch.rand((1, 64, 128, 128)),
        "unnorm_coords": torch.randn((1, 5, 2)),
        "labels": torch.ones((1, 5)),
    }

    with torch.no_grad():
        torch.onnx.export(encoder, en_inputs, "sam21_vision_encoder.onnx", opset_version=20, do_constant_folding=True, dynamo = False)
    with torch.no_grad():
        torch.onnx.export(decoder, de_inputs, "sam21_mask_decoder.onnx", opset_version=20, do_constant_folding=True, dynamo = False)

    encoder_onnx_model = onnx.load("sam21_vision_encoder.onnx")
    simplified_encoder_onnx_model, check = simplify(encoder_onnx_model)

    if check:
        onnx.save(simplified_encoder_onnx_model, "sam21_vision_encoder.onnx")

    decoder_onnx_model = onnx.load("sam21_mask_decoder.onnx")
    simplified_decoder_onnx_model, check = simplify(decoder_onnx_model)

    if check:
        onnx.save(simplified_decoder_onnx_model, "sam21_mask_decoder.onnx")


if __name__ == "__main__":
    main()
