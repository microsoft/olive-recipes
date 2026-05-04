#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------
"""  utility method to build and pre-process local dataset """
from itertools import chain
from torch.utils.data import DataLoader, Dataset
from datasets import IterableDataset, load_dataset
from transformers import default_data_collator
import torch

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
        map_kwargs["desc"] = f"Grouping texts in chunks of {self._block_size}"
        dataset = tokenized_dataset.map(
            self._group_texts,
            batched=True,
            **(map_kwargs if not isinstance(dataset, IterableDataset) else {}),
        )

        return dataset

def single_tensor_collate_fn(batch):
    # Flatten if each item is a list of tensors
    if isinstance(batch[0], list):
        batch = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in batch]
    return torch.stack(batch)

def get_local_dataset(block_size, tokenizer, json_path, key=None, batch_size = 1, percent_dataset_to_load = 100, num_samples = None, train_dataset = True):
    """
    Load and preprocess a local dataset from a JSON file.

    Args:
    block_size (int): The maximum length of sequences after tokenization.
    tokenizer: The tokenizer to use for preprocessing the text data.
    json_path (str): Path to the JSON file containing the dataset.
    key (str, optional): If provided, extracts only this key from each dataset item. Defaults to None.
    batch_size (int, optional): The batch size for the dataloader. Defaults to 1.
    percent_dataset_to_load (int, optional): Percentage of the dataset to load (1-100). Defaults to 100.
    num_samples (int, optional): If provided, limits the dataset to this many samples. Defaults to None.
    train_dataset (bool, optional): If True, loads the training split; otherwise loads the test split. Defaults to True.

    Returns:
    tuple: A tuple containing:
    - dataloader: DataLoader instance for the processed dataset
    - dataset: The original dataset dictionary with the selected split
    """
    assert 0 < percent_dataset_to_load <= 100, "percent_dataset_to_load must be greater than 0% and less than or equal to 100%"
    split = "train" if train_dataset else "test"
    class KeyExtractingDataset(Dataset):
        def __init__(self, original_dataset, key, num_samples):

            if num_samples is not None:
                self.dataset = original_dataset.select(range(min(num_samples, len(original_dataset))))
            else:
                self.dataset = original_dataset

            self.key = key

        def __getitem__(self, index):
            if self.key is None:
                return self.dataset[index]
            return self.dataset[index][self.key]

        def __len__(self):
            return len(self.dataset)

    dataset = {}
    dataset[split] = load_dataset('json', data_files=json_path, split=f'{split}[:{percent_dataset_to_load}%]')

    # Apply preprocessing
    dataset[split] = PreprocessSplit(tokenizer, block_size).preprocess(dataset[split])
    preprocessed_dataset = KeyExtractingDataset(dataset[split], key, num_samples)

    if key is not None:

        dataloader = DataLoader(preprocessed_dataset, shuffle=False, batch_size=batch_size, collate_fn=single_tensor_collate_fn)

    else:

        dataloader = DataLoader(preprocessed_dataset, shuffle=False, batch_size=1, collate_fn=default_data_collator)


    return dataloader, dataset
