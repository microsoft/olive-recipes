#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""  utility method to build and pre-process Amazon dataset """
import torch
from datasets import IterableDataset, load_dataset
from functools import partial


def get_amazon_dataset(tokenizer, processor, cache_dir, dataset_path="philschmid/amazon-product-descriptions-vlm", append_assistant_response=False):
    def _map(example, tokenizer, append_assistant_response):
        user_prompt = """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image.
        Only return description. The description should be SEO optimized and for a better mobile search experience.

        ##PRODUCT NAME##: {product_name}
        ##CATEGORY##: {category}"""

        content = user_prompt.format(product_name=example["Product Name"], category=example["Category"])

        conversations = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert product description writer for Amazon."}],
            },
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": content}],
            },
        ]

        if append_assistant_response:
            conversations.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["description"]}],
                }
            )

        return {"text": conversations}

    def _transform(example):
        inputs = tokenizer.apply_chat_template(example['text'], return_tensors="pt", return_dict=True, tokenize=True,
                                               add_generation_prompt=True)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        inputs.update({"pixel_values": torch.tensor(processor(example["image"]).pixel_values).unsqueeze(0)})
        return inputs

    dataset = load_dataset(path=dataset_path, cache_dir=cache_dir, split='train')
    dataset = dataset.map(partial(_map, tokenizer=tokenizer, append_assistant_response=append_assistant_response))
    return dataset.with_transform(_transform)
