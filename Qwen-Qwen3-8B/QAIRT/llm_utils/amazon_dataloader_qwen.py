
#!/usr/bin/env python3
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
Amazon product description dataloader for Qwen-VL.

This module provides `get_amazon_dataset`, which constructs a PyTorch
`DataLoader` for an image+text dataset compatible with Qwen-VL processors.
It formats chat-style messages (system/user/assistant), applies the processor's
chat template, handles optional image resizing, and returns a batched iterator.

Typical usage:
    dataloader, splits = get_amazon_dataset(
        dataset_path="path_or_hub_name",
        processor=processor,
        num_samples=1024,
        image_height=448,
        image_width=448,
        shuffle=True,
        cache_dir="/tmp/hf_cache",
        append_system_prompt=True,
        append_assistant_response=False,
        split="train",
    )
"""

import random
from torch.utils.data import DataLoader
from datasets import load_dataset
from qwen_vl_utils import process_vision_info


def get_amazon_dataset(
    dataset_path,
    processor,
    num_samples=None,
    image_height=None,
    image_width=None,
    shuffle=False,
    cache_dir=None,
    append_system_prompt=True,
    append_assistant_response=True,
    split="train",
):
    """
    Build a DataLoader for an Amazon-like multimodal dataset (images + text).

    This function loads a dataset (from a local path or Hugging Face hub name),
    optionally shuffles and subsamples it, and prepares a `collate_fn` that:
      * Creates chat-style messages with an optional system prompt.
      * Inserts a user prompt containing product name, category, and image.
      * Optionally appends the ground-truth assistant response (description).
      * Applies the Qwen-VL processor chat template to produce model-ready
        inputs (text tokens + image features).

    Args:
        dataset_path (str): Path or dataset identifier to load via `datasets.load_dataset`.
        processor: A Qwen-VL-compatible processor with `apply_chat_template`
            and callable interface `(text, images, videos, ...)`.
        num_samples (int, optional): If provided, trims the dataset to the first `num_samples`.
        image_height (int, optional): Target image height for resizing. Must be provided
            together with `image_width` or omitted.
        image_width (int, optional): Target image width for resizing. Must be provided
            together with `image_height` or omitted.
        shuffle (bool, optional): If True, randomly shuffles the sample order.
        cache_dir (str, optional): Cache directory for dataset loading (HF datasets).
        append_system_prompt (bool, optional): If True, prepends a system role
            message guiding the model behavior.
        append_assistant_response (bool, optional): If True, appends the ground-truth
            description as the assistant message (useful for supervised fine-tuning).
        split (str, optional): Dataset split to load (e.g., "train", "validation", "test").

    Returns:
        tuple:
            DataLoader: A PyTorch DataLoader with `batch_size=1` and the custom `collate_fn`.
            dict: A mapping of `{split_name: dataset}` for reference.

    Raises:
        AssertionError: If only one of `image_height` or `image_width` is provided.

    Notes:
        - Video inputs are currently not supported; an assertion guards against this.
        - Images are optionally resized to `(image_width, image_height)` using PIL `resize`.
        - The processor is called with `padding=False`, `truncation=False`, and `return_tensors="pt"`.
    """
    assert (image_height is None) == (image_width is None), \
        "Both image_height and image_width must be provided together or both omitted."

    dataset = load_dataset(path=dataset_path, cache_dir=cache_dir, split=split)

    if shuffle:
        random.shuffle(dataset)

    if num_samples:
        dataset = dataset[:num_samples]

    user_prompt = (
        "Create a Short Product description based on the provided <PRODUCT NAME> "
        "and <CATEGORY> and image.\nOnly return description. The description should "
        "be SEO optimized and for a better mobile search experience.\n\n"
        "<PRODUCT NAME>: {product_name}\n<CATEGORY>: {category}"
    )

    def collate_fn(sample):
        """
        Convert a single dataset sample into Qwen-VL processor inputs.

        The function expects a list with one element (due to `batch_size=1`).
        It builds a structured `messages` list following the chat format,
        applies the processor's chat template, extracts vision inputs,
        optionally resizes images, and returns a dictionary of tensors suitable
        for model consumption.

        Args:
            sample (list[dict]): A list containing a single sample with keys:
                - "Product Name" (str)
                - "Category" (str)
                - "image" (PIL.Image.Image or path-like, depending on dataset)
                - "description" (str), used if `append_assistant_response=True`

        Returns:
            dict: Model-ready inputs produced by `processor(..., return_tensors="pt")`.

        Raises:
            AssertionError: If `process_vision_info` returns non-None `video_inputs`.
        """
        sample = sample[0]

        messages = []
        if append_system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert product description writer for Amazon."}],
            })

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt.format(
                            product_name=sample["Product Name"],
                            category=sample["Category"],
                        ),
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            }
        )

        if append_assistant_response:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["description"]}],
                }
            )

        # Convert messages to text using the processor's chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Extract vision inputs (images/videos)
        image_inputs, video_inputs = process_vision_info(messages)
        assert video_inputs is None, "This processor has not been tested with video inputs"

        # Optional image resizing
        if image_height and image_width:
            image_inputs = [img.resize((image_width, image_height)) for img in image_inputs]

        # Final processor call to obtain tensors
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        return dict(inputs)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return dataloader, {f"{split}": dataset}
