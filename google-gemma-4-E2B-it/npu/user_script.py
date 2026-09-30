# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math

import numpy as np
import torch
from datasets import load_dataset
from olive.data.registry import Registry
from PIL import Image
from torch.utils.data import Dataset

DATASET_NAME = "HuggingFaceM4/the_cauldron"
PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
CELL_SIZE = PATCH_SIZE * POOLING_KERNEL_SIZE
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE


class CauldronVisionCalibrationDataset(Dataset):
    def __init__(self, images, max_patches):
        self.images = images
        self.max_patches = max_patches

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pixel_values, pixel_position_ids = preprocess_image(self.images[index], self.max_patches)
        return {
            "pixel_values": torch.from_numpy(pixel_values).unsqueeze(0),
            "pixel_position_ids": torch.from_numpy(pixel_position_ids).unsqueeze(0),
        }


@Registry.register_dataset()
def cauldron_calibration_dataset(
    subsets, samples_per_subset=16, seed=42, shuffle_buffer_size=0, max_soft_tokens=280, **kwargs
):
    """Load an even number of calibration images from each Cauldron subset."""
    del kwargs
    if max_soft_tokens <= 0:
        raise ValueError("max_soft_tokens must be greater than zero.")
    if shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size must be non-negative.")

    images = []
    for subset_index, subset in enumerate(subsets):
        dataset = load_dataset(DATASET_NAME, subset, split="train", streaming=True)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(seed=seed + subset_index, buffer_size=shuffle_buffer_size)

        subset_images = []
        for sample in dataset:
            sample_images = sample.get("images") or []
            if not sample_images:
                continue
            subset_images.append(sample_images[0].convert("RGB"))
            if len(subset_images) == samples_per_subset:
                break

        if len(subset_images) != samples_per_subset:
            raise ValueError(
                f"Cauldron subset '{subset}' yielded {len(subset_images)} usable images; "
                f"expected {samples_per_subset}."
            )
        images.extend(subset_images)

    return CauldronVisionCalibrationDataset(images, max_soft_tokens * POOLING_KERNEL_SIZE**2)


def preprocess_image(image, max_patches):
    """Create a variable-length calibration sample for the dynamic ONNX input."""
    image = image.convert("RGB")
    target_height, target_width = _get_target_size(image.height, image.width, max_patches)
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = np.transpose(pixels, (2, 0, 1))
    patches = _patchify(pixels)

    patch_height = target_height // PATCH_SIZE
    patch_width = target_width // PATCH_SIZE
    grid_x, grid_y = np.meshgrid(
        np.arange(patch_width, dtype=np.int64),
        np.arange(patch_height, dtype=np.int64),
        indexing="xy",
    )
    position_ids = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

    if not 0 < patches.shape[0] <= max_patches or patches.shape[1] != PATCH_DIM:
        raise ValueError(f"Unexpected pixel_values shape {patches.shape}; maximum patches is {max_patches}.")
    if position_ids.shape != (patches.shape[0], 2):
        raise ValueError(f"Unexpected pixel_position_ids shape {position_ids.shape}.")
    return patches, position_ids


def _get_target_size(height, width, max_patches):
    """Preserve aspect ratio while fitting a grid of 3x3 patch pooling cells."""
    target_pixels = max_patches * PATCH_SIZE**2
    scale = math.sqrt(target_pixels / (height * width))
    target_height = math.floor(scale * height / CELL_SIZE) * CELL_SIZE
    target_width = math.floor(scale * width / CELL_SIZE) * CELL_SIZE

    max_side = (max_patches // POOLING_KERNEL_SIZE**2) * CELL_SIZE
    if target_height == 0 and target_width == 0:
        raise ValueError(f"Image {height}x{width} is too small to create Gemma 4 vision patches.")
    if target_height == 0:
        target_height = CELL_SIZE
        target_width = min(math.floor(width / height) * CELL_SIZE, max_side)
    elif target_width == 0:
        target_width = CELL_SIZE
        target_height = min(math.floor(height / width) * CELL_SIZE, max_side)

    return target_height, target_width


def _patchify(pixels):
    channels, height, width = pixels.shape
    patch_height = height // PATCH_SIZE
    patch_width = width // PATCH_SIZE
    return (
        pixels.reshape(channels, patch_height, PATCH_SIZE, patch_width, PATCH_SIZE)
        .transpose(1, 3, 2, 4, 0)
        .reshape(patch_height * patch_width, PATCH_DIM)
    )
