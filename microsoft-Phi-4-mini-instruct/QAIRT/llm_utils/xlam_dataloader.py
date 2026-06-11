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

class xLAMDataset():
    def __init__(self, tokenizer, block_size, batch_size, return_train_dataset=True, return_test_dataset=False):
        self._tokenizer = copy.deepcopy(tokenizer)
        self._block_size = block_size
        self._batch_size = batch_size
        self._return_train_dataset = return_train_dataset
        self._return_test_dataset = return_test_dataset

    def get_xlam_dataloader(self, path):
        tokenizer = self._tokenizer
        block_size = self._block_size
        batch_size = self._batch_size

        dataset = load_dataset(path)

        # Process both train and test data
        processed_datasets = {}
        for split, data in dataset.items():
            split_data = {'input_ids': [], 'attention_mask': []}
            for row_tmp in data:
                prompt, answer = self.process_xLAM_processing(row_tmp)
                if self._return_test_dataset:
                    row_text = prompt
                if self._return_train_dataset:
                    row_text = prompt + answer + tokenizer.eos_token
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

    def process_xLAM_processing(self, row):
        tokenizer = self._tokenizer

        TEST_SYSTEM_PROMPT_FOR_CHAT_MODEL ="You are an expert in composing functions. You are given a question and a set of possible functions.\nBased on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections.\nIf you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\nYou SHOULD NOT include any other text in the response.\nHere is a list of functions in JSON format that you can invoke.\n{functions}"

        SYSTEM_PROMPT_FOR_CHAT_MODEL = """
            You are an expert in composing functions. You are given a question and a set of possible functions.
            Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
            If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
            also point it out. You should only return the function call in tools call sections.
            """

        USER_PROMPT_FOR_CHAT_MODEL = """
            Questions:{user_prompt}\nHere is a list of functions in JSON format that you can invoke:\n{functions}.
            Should you decide to return the function call(s),Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n
            NO other text MUST be included.
        """
        def _format_prompt(prompt, function):

            if self._return_test_dataset:
                raw_prompt = [{"role": "system", "content": TEST_SYSTEM_PROMPT_FOR_CHAT_MODEL.format(functions=function)},{"role": "user", "content": prompt}]
                return tokenizer.apply_chat_template(raw_prompt, tokenize=False,add_generation_prompt=True)
            
            if self._return_train_dataset:
                conversations = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{SYSTEM_PROMPT_FOR_CHAT_MODEL}<|eot_id|><|start_header_id|>user<|end_header_id|>{USER_PROMPT_FOR_CHAT_MODEL.format(user_prompt=prompt, functions=str(function))}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                return conversations

        def convert_json_to_function_calls(json_str):
            # Parse the JSON string
            data = json.loads(json_str)

            # Create the desired output format
            output_list = []
            for item in data:
                func_name = item['name']
                args = ', '.join([f"{k}={json.dumps(v)}" for k, v in item['arguments'].items()])
                output_list.append(f"{func_name}({args})")

            # Convert the list to a string
            output_str = f"[{', '.join(output_list)}]"
            return output_str

        prompt = _format_prompt(row["query"], row["tools"])
        answer = convert_json_to_function_calls(row['answers'])
        return prompt, answer