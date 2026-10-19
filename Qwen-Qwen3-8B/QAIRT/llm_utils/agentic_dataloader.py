
#!/usr/bin/env python3
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

def get_agentic_dataset(datafile, tokenizer, max_length, cache_dir=None):
    """
    Load and preprocess an agentic dataset for language model training/inference.

    This function loads a JSON dataset, applies text preprocessing including chat template
    formatting and tokenization, and returns a DataLoader ready for model consumption.

    Args:
        datafile (str): Path to the JSON data file containing the dataset.
            Expected format: JSON file with entries containing at least a "prompt" field.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer instance used for
            tokenizing the text data. Should have `apply_chat_template` method.
        max_length (int): Maximum sequence length for tokenization. Sequences will be
            padded or truncated to this length.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (DataLoader): PyTorch DataLoader for the training dataset
              with batch_size=1, configured with default_data_collator.
            - dataset (DatasetDict): The processed HuggingFace dataset dictionary containing
              the 'train' split with tokenized data.

    Dataset Processing Pipeline:
        1. Loads JSON data from the specified file
        2. Applies chat template formatting (if needed via apply_chat_template)
        3. Converts prompt text with unicode escape decoding
        4. Removes unnecessary columns, keeping only the processed text
        5. Tokenizes text with padding to max_length
        6. Removes intermediate 'text' column
        7. Creates DataLoader with batch_size=1

    Expected Input Format:
        The JSON file should contain entries with:
        - "prompt": The input prompt text (required)
        - "generated_answer": The expected answer (optional, removed during processing)
        - Other fields will be automatically removed during preprocessing

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataloader, dataset = get_agentic_dataset(
        ...     datafile="data/prompts.json",
        ...     tokenizer=tokenizer,
        ...     max_length=512
        ... )
        >>> for batch in dataloader:
        ...     outputs = model(**batch)

    Note:
        - The function assumes a 'train' split in the dataset
        - All columns except 'input_ids' and tokenizer outputs are removed
        - Unicode escape sequences in prompts are decoded (e.g., \\n becomes newline)
        - The DataLoader uses batch_size=1 and no shuffling by default
    """
    prompt_key = "prompt"
    answer_key = "generated_answer"

    dataset = load_dataset(
        'json',
        data_files=[datafile],
        cache_dir=cache_dir
    )

    def apply_chat_template(example):
        example["prompt"] = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return example

    def add_text(example):
        # apply chat template if needed
        example["text"] = example[prompt_key].encode().decode('unicode_escape')
        no_needs = []
        for key in example.keys():
            if key != prompt_key and key != "text":
                no_needs.append(key)
        for key in no_needs:
            del example[key]
        del example[prompt_key]
        return example

    def tokenize(example):
        example = tokenizer(example["text"], return_tensors="pt", padding="max_length", max_length=max_length)
        example["input_ids"] = example["input_ids"][0]
        example["attention_mask"] = example["attention_mask"][0]
        return example

    dataset = dataset.map(add_text)
    dataset = dataset.map(tokenize)
    dataset["train"] = dataset["train"].remove_columns("text")

    train_dataloader = DataLoader(
        dataset=dataset['train'],
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=default_data_collator,
    )

    return train_dataloader, dataset
