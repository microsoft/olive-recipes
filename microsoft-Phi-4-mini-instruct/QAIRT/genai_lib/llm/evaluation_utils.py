#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides evaluation utilities for LLM Lib """

import contextlib
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
from torch.utils._pytree import tree_map


def change_tensor_device_placement(input_data, device: torch.device):
    """
    Change the tensor_data's device placement

    :param input_data: torch.tensor , list of torch.tensors, or tuple of torch.tensors
    :param device: device
    :return: tensor_data with modified device placement

    Duplicated code with AIMET_Torch to remove dependency
    """
    return tree_map(
        lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, input_data
    )

@contextlib.contextmanager
def _place_model_in_eval_mode(model):
    '''Temporarily switch to evaluation mode.'''
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()
def llm_compute_loss_from_logits(outputs, labels):
    '''
    This function computes the loss from the logits and the labels passed
    '''
    #Get the outputs and move it to CPU. Assumes that index 0 is logits as
    lm_logits = outputs[0].cpu()
    shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

    #Compute the loss
    loss_fn = CrossEntropyLoss()
    neg_log_likelihood = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return neg_log_likelihood

@torch.no_grad()
def llm_evaluate_ppl(model, encoded_dataset, max_length, stride=2048) -> float:
    '''
    This function takes in an encoded dataset string and computes ppl
    params:
    model: the model to evaluate
    encoded_dataset: the encoded dataset
    max_length: represents the max length of inputs model can take in
    stride: the stride used for sliding window over the dataset
    '''
    nlls = []
    prev_end_loc = 0
    seq_len = encoded_dataset.input_ids.size(1)
    device = model.device
    with _place_model_in_eval_mode(model):

        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encoded_dataset.input_ids[:, begin_loc:end_loc].to(device)
            labels = input_ids.clone()
            labels[:, :-trg_len] = -100

            #Labels are not passed into the model. Loss computation happens outside the mode
            outputs = model(input_ids)

            nlls.append(llm_compute_loss_from_logits(outputs,labels))
            del outputs
            #Set up variables for the next batch of data
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    ppl = torch.exp(torch.stack(nlls).mean())
    return float(ppl)

@torch.no_grad()
def llm_evaluate_ppl_with_dataloader(model, dataloader, num_batches=None, model_forward_kwargs={}):
    '''
    This function takes in a dada loader and a model and computes ppl score
    params:
    model: the model to evaluate
    dataloader: dataset loader
    num_batches: number of batches to run evaluation on
    '''
    num_batches = num_batches if num_batches else len(dataloader)
    nlls=[]
    device=model.device
    model_forward_kwargs = change_tensor_device_placement(model_forward_kwargs, device)

    for batch_id, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Evaluating")):
        if batch_id >= num_batches:
            break
        if "inputs_embeds" in batch:
            batch["input_ids"] = batch["labels"]
            batch["inputs_embeds"] = batch["inputs_embeds"].to(device)
            outputs = model(inputs_embeds=batch["inputs_embeds"], **model_forward_kwargs)
        else:
            batch["input_ids"] = batch["input_ids"].to(device)
            outputs = model(input_ids=batch["input_ids"], **model_forward_kwargs)

        nlls.append(llm_compute_loss_from_logits(outputs, batch["input_ids"]))
        del outputs
    ppl = torch.exp(torch.stack(nlls).mean())
    return float(ppl)