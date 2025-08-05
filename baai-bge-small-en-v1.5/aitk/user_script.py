# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import datasets
import mteb
import numpy as np
import torch
from transformers import AutoTokenizer

from olive.constants import Framework
from olive.data.registry import Registry
from olive.model import OliveModelHandler

# -------------------------------------------------------------------------
# Common Dataset
# -------------------------------------------------------------------------

seed = 0
# seed everything to 0 for reproducibility, https://pytorch.org/docs/stable/notes/randomness.html
# do not set random seed and np.random.seed for aml test, since it will cause aml job name conflict
torch.manual_seed(seed)
# the following are needed only for GPU
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set max sequence length
MAX_SEQ_LENGTH = 128

# define the tokenizer for BGE model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
VOCAB_SIZE = len(tokenizer)

# set default input
default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)

# define model inputs
model_inputs = {
    "input_ids": default_input,
    "attention_mask": default_input,
    "token_type_ids": default_input,
}

# capture input names
INPUT_NAMES = list(model_inputs)


class OliveEncoder:
    def __init__(self, model, session):
        self.model = model
        self.session = session
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    def encode(self, corpus: list, **kwargs):
        model_output = None
        if self.model.framework == Framework.ONNX:
            encoded_input = self.tokenizer(
                corpus, padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True, return_tensors="np"
            )
            # batch_size is 1 for static model
            model_outputs = []
            for i in range(len(corpus)):
                model_inputs = {
                    "input_ids": encoded_input.input_ids[i : i + 1, :].astype(np.int64),
                    "attention_mask": encoded_input.attention_mask[i : i + 1, :].astype(np.int64),
                    "token_type_ids": encoded_input.token_type_ids[i : i + 1, :].astype(np.int64),
                }
                model_output = self.model.run_session(self.session, model_inputs)[0]
                model_outputs.append(model_output[0])
            model_output = np.array(model_outputs)
        elif self.model.framework == Framework.PYTORCH:
            encoded_input = self.tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
            model_inputs = {
                "input_ids": encoded_input.input_ids,
                "attention_mask": encoded_input.attention_mask,
                "token_type_ids": encoded_input.token_type_ids,
            }
            with torch.no_grad():
                model_output = self.model.run_session(self.session, model_inputs)
            model_output = model_output.last_hidden_state.numpy()
        # select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.
        return model_output[:, 0, :]


def eval_accuracy(model: OliveModelHandler, device, execution_providers, tasks=None):
    """Evaluate accuracy using MTEB (Massive Text Embedding Benchmark) for standardized evaluation."""
    sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)
    
    # Use default tasks if none provided
    if tasks is None:
        tasks = ["Banking77Classification"]  # Default to Banking77 for BGE model evaluation
    
    evaluation = mteb.MTEB(tasks=tasks)
    olive_encoder = OliveEncoder(model, sess)
    results = evaluation.run(olive_encoder, output_folder=None)
    
    # Return the main score from the first task
    return results[0].scores["test"][0]["main_score"]


@Registry.register_dataset()
def bge_small_en_dataset(data_name, split, max_samples):
    # load the raw wikipedia dataset for tuning. Load just 300 examples for speed.
    raw_dataset = datasets.load_dataset(data_name, "20220301.en", split=split, trust_remote_code=True)
    
    # Apply max_samples limit after loading
    if max_samples:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))

    def _preprocess_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )

    # preprocess the dataset
    return raw_dataset.map(_preprocess_fn, batched=True, batch_size=1)


def custom_transform_func(data_item):
    return {
        name: np.asarray([np.array([g.flatten() for g in data_item[name]]).flatten()], dtype=np.int64)
        for name in INPUT_NAMES
    }


def custom_example_func():
    vocab_size = VOCAB_SIZE
    batch_size = 1
    sequence_length = MAX_SEQ_LENGTH

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # Generate random attention_mask (1s for actual tokens, 0s for padding)
    attention_mask = default_input

    # Generate random token_type_ids (0 for sentence 1, 1 for sentence 2)
    token_type_ids = default_input

    return [input_ids, attention_mask, token_type_ids]