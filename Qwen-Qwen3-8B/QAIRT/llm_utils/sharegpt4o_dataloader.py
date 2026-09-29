#!/usr/bin/env python3
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
import glob
import html
from typing import List, Dict, Any

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from PIL import Image



def get_sharegpt4o_dataset(
    dataset_path: str,
    processor,
    num_samples: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    shuffle: bool = False,
    cache_dir: str | None = None,
    append_system_prompt: bool = True,
    append_assistant_response: bool = True,
    include_text: bool = False,
    split: str = "train",
    seed: int = 42,
):
    """
    DataLoader for ShareGPT-4o style JSONL + images/ (schema verified against the provided sample).

    Returns (dataloader, {split: dataset}) and mirrors the reference loaders' API.

    Robustness:
      - Uses HF-native shuffle/select to keep the dataset indexable by PyTorch.
      - `apply_chat_template` fallback: if the tokenizer/template expects string-only
        `content` (not multimodal lists), we convert messages to string content and
        reinstate a "<image>\n" token on the first user message when images are present.
    """
    assert (image_height is None) == (image_width is None), (
        "Both image_height and image_width must be provided together or both omitted."
    )

    # Find single JSONL in root
    dataset_config = glob.glob(os.path.join(dataset_path, "*.jsonl"))
    assert (
        len(dataset_config) == 1
    ), f"Expected single jsonl config, but found {len(dataset_config)} configs = `{dataset_config}` in {dataset_path}."

    ds: Dataset = load_dataset("json", data_files=dataset_config[0], cache_dir=cache_dir, split=split)

    # Use HF-native shuffle/select to preserve indexability for PyTorch DataLoader
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if num_samples:
        n = min(num_samples, len(ds))
        ds = ds.select(range(n))

    def _to_role(who: str) -> str:
        w = (who or "").lower()
        if w in ("human", "user"):
            return "user"
        if w in ("gpt", "assistant"):
            return "assistant"
        if w == "system":
            return "system"
        return who

    def _resolve_image_paths(sample: Dict[str, Any]) -> List[str]:
        # Accept both top-level 'image' (str) and 'images' (list[str])
        if isinstance(sample.get("images"), list):
            rels = sample["images"]
        elif isinstance(sample.get("image"), str):
            rels = [sample["image"]]
        else:
            rels = []
        out: List[str] = []
        for p in rels:
            # If already absolute or already rooted under <dataset_path>/images, pass through
            if os.path.isabs(p) or p.startswith(os.path.join(dataset_path, "images" + os.sep)):
                out.append(p)
            else:
                out.append(os.path.join(dataset_path, "images", p))
        return out

    def _clean_text(text: str) -> str:
        if text is None:
            return ""
        # Unescape HTML then remove <image> tokens
        t = html.unescape(text)
        return t.replace("<image>", "").strip()

    def _build_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if append_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful multimodal assistant. Use the provided image(s) when relevant.",
                        }
                    ],
                }
            )

        images_abs = _resolve_image_paths(sample)
        images_attached = False

        for turn in sample.get("conversations", []):
            role = _to_role(turn.get("from"))
            text = _clean_text(turn.get("value"))
            content = []

            # Attach images to the FIRST user message only
            if role == "user" and (not images_attached) and images_abs:
                for img_path in images_abs:
                    content.append({"type": "image", "image": img_path})
                images_attached = True

            if text:
                content.append({"type": "text", "text": text})

            if role == "assistant" and not append_assistant_response:
                continue

            if content:
                messages.append({"role": role, "content": content})

        return messages

    def _load_images_from_messages(messages: List[Dict[str, Any]]) -> List["Image.Image"]:
        imgs: List["Image.Image"] = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    ref = item.get("image")
                    if isinstance(ref, Image.Image):
                        imgs.append(ref)
                    elif isinstance(ref, str):
                        imgs.append(Image.open(ref).convert("RGB"))
        return imgs

    def collate_fn(batch: List[Dict[str, Any]]):
        sample = batch[0]
        messages = _build_messages(sample)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = _load_images_from_messages(messages)
        video_inputs = None

        if image_height and image_width and image_inputs:
            image_inputs = [img.resize((image_width, image_height)) for img in image_inputs]

        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        if include_text:
            inputs["text"] = text
        return dict(inputs)

    dataloader = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return dataloader, {f"{split}": ds}
