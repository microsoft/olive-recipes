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


def get_amazon_dataset(tokenizer, processor, cache_dir, dataset_path="philschmid/amazon-product-descriptions-vlm", append_assistant_response=False, append_system_prompt=True):
    def _map(example, tokenizer, append_assistant_response, append_system_prompt):
        user_prompt = "Create a Short Product description based on the provided <PRODUCT NAME> and <CATEGORY> and image. \nOnly return description. The description should be SEO optimized and for a better mobile search experience.\n\n<PRODUCT NAME>: {product_name}\n<CATEGORY>: {category}"

        content = user_prompt.format(product_name=example["Product Name"], category=example["Category"])

        conversations = []

        if append_system_prompt:
            conversations.append({
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert product description writer for Amazon."}],
            })

        conversations.append({
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": content}],
            })

        if append_assistant_response:
            conversations.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "image": example["image"]
                }
            )

        return {"text": conversations}

    def _transform(example):
        inputs = tokenizer.apply_chat_template(example['text'], return_tensors="pt", return_dict=True, tokenize=True,
                                               add_generation_prompt=True)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        inputs.update({"pixel_values": torch.tensor(processor(example["image"]).pixel_values).unsqueeze(0)})

        # Support custom image tokens by decoding, replacing text, and re-encoding
        if hasattr(processor, "image_token"):
            inputs['input_ids'] = torch.tensor(tokenizer.encode(tokenizer.decode(inputs['input_ids'][0][0][1:]).replace("<|image|>", processor.image_token))).unsqueeze(0).unsqueeze(0)
            inputs['attention_mask'] = torch.ones((inputs['input_ids'].shape[-3], inputs['input_ids'].shape[-2], inputs['input_ids'].shape[-1]), dtype = torch.long, device = inputs['input_ids'].device)
        return inputs

    dataset = load_dataset(path=dataset_path, cache_dir=cache_dir, split='train')
    dataset = dataset.map(partial(_map, tokenizer=tokenizer, append_assistant_response=append_assistant_response, append_system_prompt=append_system_prompt))
    return dataset.with_transform(_transform)
