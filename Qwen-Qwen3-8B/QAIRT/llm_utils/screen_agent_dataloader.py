#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import random
from torch.utils.data import DataLoader
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import glob

def get_screen_agent_dataset(dataset_path,
                             processor,
                             num_samples=None,
                             image_height=None,
                             image_width=None,
                             shuffle=False,
                             cache_dir=None,
                             append_system_prompt=True,
                             append_assistant_response=True,
                             split='train'):

    assert (image_height is None) == (image_width is None), \
        "Both image_height and image_width must be provided together or both omitted."

    dataset_config = glob.glob(os.path.join(dataset_path, "*.jsonl"))
    assert len(dataset_config) == 1, f"Expected single jsonl config, but found {len(dataset_config)} configs = `{dataset_config}` in {dataset_path}."
    dataset = load_dataset("json", data_files=dataset_config[0], cache_dir=cache_dir, split=split)
    if shuffle:
        random.shuffle(dataset)
    if num_samples:
        dataset = dataset[:num_samples]

    def collate_fn(batch):
        batch = batch[0]
        messages = []

        if append_system_prompt:
            messages.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an expert AI assistant specializing in UI analysis.\nYour task is to extract specific, key UI elements from a given screenshot and structure them according to the provided JSON schema."}
                ],
            })

        messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(dataset_path, batch["image"]),
                    },
                ],
            })

        if append_assistant_response:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": batch["response_json_text_format"]},
                ],
            })

        # Convert messages to text
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Get image inputs from process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        assert video_inputs is None, "This processor does not support video inputs"

        # Resize images if specific size provided by user
        if image_height and image_width:
            image_inputs = [img.resize((image_width, image_height)) for img in image_inputs]

        # Process text and images
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        inputs['text'] = text

        return dict(inputs)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
    return dataloader, {f"{split}": dataset}
