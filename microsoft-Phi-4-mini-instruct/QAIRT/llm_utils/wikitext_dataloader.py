# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
#
# This file contains certain notices of software components included with the
# software that Qualcomm Technologies, Inc. ("QTI") is required to provide you.
# Except where prohibited by the open source license, the content of this file is
# provided solely to satisfy QTI's attribution and notice requirement; your use of
# these software components together with the QTI software ("Software") is subject
# to the terms of your license from QTI. Compliance with all copyright laws and
# software license agreements included in the notice section of this file are the
# responsibility of the user. Except as may be granted by separate express written
# agreement, this file provides no license to any patents, trademarks, copyrights,
# or other intellectual property of Qualcomm Incorporated or any of its
# subsidiaries.
#
# Software provided with this notice is NOT A CONTRIBUTION to any open source
# project. If alternative licensing is available for any of the components with
# licenses or attributions provided below, a license choice is made for receiving
# such code by QTI.

# Copyright (c) 2023 Qualcomm Technologies, Inc. All rights reserved.

# Qualcomm is a trademark of Qualcomm Incorporated, registered in the United
# States and other countries. All Qualcomm Incorporated trademarks are used with
# permission. Other products and brand names may be trademarks or registered
# trademarks of their respective owners.
#
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""  utility method to evaluate perplexity score on WikiText """
from itertools import chain
from torch.utils.data import DataLoader, Dataset
from datasets import IterableDataset, load_dataset
from transformers import default_data_collator


class CustomDataset(Dataset):
    """
    Dataset for GPTQ-preprocessed tokens
    """
    def __init__(self, tokens, block_size=2048):
        self.full_tokens = tokens
        self.block_size = block_size
        self._len = len(tokens["input_ids"][0]) // block_size

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = (idx+1) * self.block_size

        input_ids = self.full_tokens["input_ids"][0, start_idx:end_idx]
        labels = input_ids.clone()
        attn_mask = self.full_tokens["attention_mask"][0, start_idx:end_idx]
        output = {"input_ids": input_ids,
                  "attention_mask": attn_mask,
                  "labels": labels}
        return output

def _get_column_names(dataset):
    if hasattr(dataset, "column_names"):
        return dataset.column_names
    else:
        return next(iter(dataset.take(1))).keys()


def get_column_name(dataset):
    column_names = _get_column_names(dataset)
    if "text" in column_names:
        return "text"
    else:
        return column_names[0]

class PreprocessGptqSplit:
    def __init__(self, tokenizer, block_size, add_special_tokens=True):
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._add_special_tokens = add_special_tokens

    def preprocess(self, dataset):
        column_name = get_column_name(dataset)
        tokens = self._tokenizer("\n\n".join(dataset[column_name]), return_tensors="pt", add_special_tokens=self._add_special_tokens)
        dataset = CustomDataset(tokens, self._block_size)
        return dataset

class PreprocessSplit:
    def __init__(self, tokenizer, block_size, column_name="text", add_special_tokens=True):
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._column_name = column_name
        self._add_special_tokens = add_special_tokens

    def _tokenize_fn(self, examples):
        return self._tokenizer(examples[self._column_name], return_token_type_ids=False, add_special_tokens=self._add_special_tokens)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def _group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self._block_size:
            total_length = (total_length // self._block_size) * self._block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self._block_size] for i in range(0, total_length, self._block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess(self, dataset):
        map_kwargs = {
            "num_proc": None,
            "load_from_cache_file": True,
            "desc": "Running tokenizer on dataset",
        }

        tokenized_dataset = dataset.map(
            self._tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            **(map_kwargs if not isinstance(dataset, IterableDataset) else {}),
        )

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        print(f"[Load Dataset]grouped train dataset")
        map_kwargs["desc"] = f"Grouping texts in chunks of {self._block_size}"
        dataset = tokenized_dataset.map(
            self._group_texts,
            batched=True,
            **(map_kwargs if not isinstance(dataset, IterableDataset) else {}),
        )

        return dataset

def get_wiki_dataset(block_size, tokenizer, cache_dir):
    dataset = {}
    dataset['train'] = load_dataset(path='wikitext',
                                    name='wikitext-2-raw-v1',
                                    cache_dir=cache_dir,
                                    split='train')
    dataset['train'] = PreprocessSplit(tokenizer, block_size).preprocess(dataset['train'])

    dataset['test'] = load_dataset(path='wikitext',
                                   name='wikitext-2-raw-v1',
                                   cache_dir=cache_dir,
                                   split='test')
    dataset['test'] = PreprocessGptqSplit(tokenizer, block_size).preprocess(dataset['test'])

    train_dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=1, collate_fn=default_data_collator)
    test_dataloader = DataLoader(dataset['test'], shuffle=False, batch_size=1, collate_fn=default_data_collator)

    return train_dataloader, test_dataloader, dataset
