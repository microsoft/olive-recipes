# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from olive.data.registry import Registry
from PIL import Image
from tokenizers import Tokenizer
from torch.utils.data import Dataset

DATASET_NAME = "HuggingFaceM4/the_cauldron"
DATASET_REPO = "Salesforce/wikitext"
DATASET_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
CELL_SIZE = PATCH_SIZE * POOLING_KERNEL_SIZE
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE
NUM_KV_LAYERS = 15
HEAD_DIM = 256
GLOBAL_LAYERS = frozenset({4, 9, 14})
GLOBAL_HEAD_DIM = 512


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
    from datasets import load_dataset

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


class DecoderCalibrationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


@Registry.register_dataset()
def wikitext_decoder_calibration_dataset(
    embedding_model_path,
    tokenizer_path=None,
    num_samples=32,
    seq_len=64,
    past_len=64,
    **kwargs,
):
    """Build decoder calibration samples from WikiText 2."""
    del kwargs
    if num_samples <= 0 or seq_len <= 0 or past_len <= 0:
        raise ValueError("num_samples, seq_len, and past_len must all be greater than zero.")

    tokenizer_path = tokenizer_path or Path(embedding_model_path).parents[1] / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    parquet_path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    texts = pd.read_parquet(parquet_path, engine="fastparquet")["text"].tolist()

    ids = []
    needed = num_samples * seq_len
    for text in texts:
        text = text.strip()
        if len(text) < 200:
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False).ids)
        if len(ids) >= needed * 4:
            break
    if len(ids) < needed:
        raise ValueError(f"Not enough calibration text: have {len(ids)} tokens, need {needed}.")

    stride = len(ids) // num_samples
    windows = [ids[index * stride : index * stride + seq_len] for index in range(num_samples)]

    session = ort.InferenceSession(embedding_model_path, providers=["CPUExecutionProvider"])
    empty_features = np.zeros((0, 1536), dtype=np.float32)
    empty_kv = {
        layer: torch.zeros(
            (1, 1, past_len, GLOBAL_HEAD_DIM if layer in GLOBAL_LAYERS else HEAD_DIM),
            dtype=torch.float16,
        )
        for layer in range(NUM_KV_LAYERS)
    }

    samples = []
    for window in windows:
        embeds, per_layer = session.run(
            ["inputs_embeds", "per_layer_inputs"],
            {
                "input_ids": np.asarray([window], dtype=np.int64),
                "image_features": empty_features,
                "audio_features": empty_features,
            },
        )
        sample = {
            "inputs_embeds": torch.from_numpy(embeds.astype(np.float16)),
            "per_layer_inputs": torch.from_numpy(per_layer.astype(np.float16)),
            # GroupQueryAttention reads this as seqlens_k, which is
            # total_sequence_length - 1, despite the graph input name.
            "past_seq_len": torch.full((1, 1), seq_len - 1, dtype=torch.int32),
            "total_seq_len": torch.tensor(seq_len, dtype=torch.int32),
        }
        for layer in range(NUM_KV_LAYERS):
            sample[f"past_key_values.{layer}.key"] = empty_kv[layer].clone()
            sample[f"past_key_values.{layer}.value"] = empty_kv[layer].clone()
        samples.append(sample)

    return DecoderCalibrationDataset(samples)
