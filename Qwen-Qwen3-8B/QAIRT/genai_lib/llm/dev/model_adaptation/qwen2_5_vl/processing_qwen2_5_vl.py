#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Custom image pre-processor for Qwen2.5 VL """

try:
    from qwen_vl_utils.vision_process import smart_resize, fetch_video, to_rgb
except ImportError as e:
    raise ImportError(
        "The 'qwen_vl_utils' package is required to use 'QcQwen2_5_VLProcessor'. "
        "Please install it using 'pip install qwen-vl-utils' or ensure it's available in your environment."
    ) from e

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor


class QcQwen2_5_VLProcessor(Qwen2_5_VLProcessor):

    """
    Custom image and video pre-processor for Qwen2.5 VL, modifying the Huggingface version
    to be aligned with Qwen's reference implementation.
    https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
    https://github.com/QwenLM/Qwen3-VL/blob/0dcc180d854f4b132f8059b10bbf0b5fd5dae9ed/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L120-L141

    Args:
        patch_size (int): Size of each image patch (default: 14).
        spatial_merge_size (int): Spatial merge factor (default: 2).
        max_image_tokens (int or None): Upper bound on the number of image tokens.
        min_image_tokens (int or None): Lower bound on the number of image tokens.
        *args, **kwargs: Additional arguments passed to the base processor.

    Attributes:
        factor (int): The product of patch size and spatial merge size, used to compute
                    the resizing factor for images.
        min_pixels (int or None): Minimum number of pixels (height × width) allowed for
                                resized images, derived from `min_image_tokens`.
        max_pixels (int or None): Maximum number of pixels (height × width) allowed for
                                resized images, derived from `max_image_tokens`.
    """

    def __init__(self,
                 *args,
                 patch_size=14,
                 spatial_merge_size=2,
                 max_image_tokens=None,
                 min_image_tokens=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = patch_size * spatial_merge_size
        self.min_pixels = min_image_tokens * self.factor**2 if min_image_tokens else None
        self.max_pixels = max_image_tokens * self.factor**2 if max_image_tokens else None

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        **kwargs,
    ):
        """
        Preprocess images and videos with optional token count constraints.

        - Images are converted to RGB and resized using `smart_resize` to ensure the
          resulting token count (based on patching and merging) falls within the
          specified range.
        - Videos are loaded using `fetch_video`.

        Args:
            images (List[PIL.Image] or None): List of input images to preprocess.
            text (str or None): Optional text input.
            videos (List[str] or None): List of video file paths to preprocess.
            **kwargs: Additional keyword arguments passed to the base processor.

        Returns:
            dict: A dictionary of processed inputs compatible with the Qwen2.5 VL model.
        """

        if images is not None:
            images_ = []
            for image in images:
                image = to_rgb(image)
                width, height = image.size
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.factor,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = image.resize((resized_width, resized_height))
                images_.append(image)
            images = images_
        if videos is not None:
            videos_ = []
            for video in videos:
                video = fetch_video({"video": video})
                videos_.append(video)
            videos = videos_

        return super().__call__(images=images, text=text, videos=videos, **kwargs)
