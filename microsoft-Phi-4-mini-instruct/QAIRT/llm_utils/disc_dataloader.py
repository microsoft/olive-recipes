#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
from itertools import chain

from datasets import load_dataset, Dataset
from transformers import default_data_collator
from torch.utils.data import DataLoader
import copy

class DiSCDataset():
    def __init__(self, tokenizer, block_size, batch_size, return_train_dataset=True, return_test_dataset=False):
        self._tokenizer = copy.deepcopy(tokenizer)
        self._block_size = block_size
        self._batch_size = batch_size
        self._return_train_dataset = return_train_dataset
        self._return_test_dataset = return_test_dataset

    def get_disc_dataloader(self, path):
        tokenizer = self._tokenizer
        block_size = self._block_size
        batch_size = self._batch_size

        dataset = load_dataset(path)
        categor_list = ['grade_elementary', 'length_long']

        # Process both train and test data
        processed_datasets = {}
        for split, data in dataset.items():
            split_data = {'input_ids': [], 'attention_mask': []}
            for row_tmp in data:
                if row_tmp['category'] in categor_list:
                    chat, _, _ = self.process_style_adapt_template(row_tmp, False)
                    row_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=self._return_test_dataset)
                    tokenizer.pad_token = tokenizer.eos_token
                    row1 = tokenizer(row_text, return_tensors="pt", truncation=True, max_length=block_size, padding="max_length")
                    split_data['input_ids'].append(row1['input_ids'][0])
                    split_data['attention_mask'].append(row1['attention_mask'][0])

            processed_datasets[split] = Dataset.from_dict(split_data)

        collate_fn = default_data_collator

        train_dataloader = DataLoader(
            processed_datasets['train'], shuffle=False,
            batch_size=batch_size,
            collate_fn=collate_fn,
        ) if self._return_train_dataset else None
        
        test_dataloader = DataLoader(
            processed_datasets['train'], shuffle=False,
            batch_size=batch_size,
            collate_fn=collate_fn,
        ) if self._return_test_dataset else None

        return train_dataloader, test_dataloader, processed_datasets

    def process_style_adapt_template(self, row, reverse):
        system_prompt = 'You are a helpful style adapter. Produce one unique rephrasing of the following content using a different style.'
        if reverse:
            user_content = row['generation']
            assistant_content = row['original']
        else:
            user_content = row['original']
            assistant_content = row['generation']

        if self._return_train_dataset:
            chat = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"{user_content}"},
                    {'role': 'assistant', 'content': f"{assistant_content}"}]
        if self._return_test_dataset :
            chat = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"{user_content}"}]
        #row_text = self.cfg['tokenizer'].apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return chat, user_content, assistant_content