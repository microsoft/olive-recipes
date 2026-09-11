#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""  utility method to build and pre-process WikiText dataset """
import os
import torch
from itertools import chain
from torch.utils.data import DataLoader, Dataset
from datasets import IterableDataset, load_dataset
from transformers import default_data_collator
from PIL import Image as PILImage
from transformers.feature_extraction_utils import BatchFeature

def _convert_one_conversation(conversation: list[dict[str, str]]) -> dict[str, str | list[dict]]:
        """Convert the given conversation to LLaVA style.

        Examples:

            >>> conversation = {"from": "human", "value": "<image>What are the colors of the bus in the image?"}
            >>> LlavaConversationProcessor._convert(conversation)
            {
                'role': 'user',
                'content': [{'type': 'image'}, {'type': 'text', 'text': 'What are the colors of the bus in the image?'}]
            }
            >>> conversation = {"from": "gpt", "value": "The bus in the image is white and red."}
            >>> _convert(conversation)
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'The bus in the image is white and red.'}]
            }
        """
        who = conversation.get("from")
        match who:
            case "human":
                role = "user"
            case "gpt":
                role = "assistant"
            case _:
                raise ValueError(f"Unknown role: {who}")

        text = conversation.get("value")

        if "<image>" in text:
            has_image = True
            text = text.replace("<image>", "")
        else:
            has_image = False

        return {
            "role": role,
            "content": (
                [{"type": "image"}, {"type": "text", "text": text}] if has_image else [{"type": "text", "text": text}]
            ),
        }

def get_llava_dataset(tokenzier, processor, data_files, dataset_path, cache_dir):
    def _map(examples):
        examples['text'] = [_convert_one_conversation(conversation=conversation) for conversation in
                            examples['conversations']]
        return examples

    def _load_image_and_tokenize(example):
        inputs = tokenzier.apply_chat_template(example['text'], add_generation_prompt=True, tokenize=True,
                                               return_tensors="pt", return_dict=True)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        inputs.update({"pixel_values": torch.tensor(processor(PILImage.open(fp=os.path.join(dataset_path, example["image"][0]))).pixel_values).unsqueeze(0)})
        return inputs

    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split='train')
    dataset = dataset.map(_map)
    return dataset.with_transform(_load_image_and_tokenize)
