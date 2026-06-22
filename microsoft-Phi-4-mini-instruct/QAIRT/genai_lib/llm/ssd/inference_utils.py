#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides utilities that are needed for token generation with SSD """
import torch
from genai_lib.llm.utils import _concat
import functools
from transformers import TopKLogitsWarper
import numpy as np
from copy import deepcopy
DEFAULT_STOP_SEQUENCES = ["###", "<|endoftext|>", "<|begin_of_text|>", \
                          "<|eot_id|>", "<|end_of_text|>",  "<|im_start|>", \
                          "<|im_end|>", "<|end|>", "<\s>", "</s>", "<s>", \
                          "\n\n\n\n\n"]

def llm_generate_forecast_tokens(num_draft_tokens, num_forecast_per_token, vocab_size, batch_size = 1, device = "cpu"):
    '''
    This API is responsible for generating the num_forecast_per_token tokens. The token IDs are determined assuming they are appended to the right of the existing vocabulary.
    For SSD, the forecast tokens are appended to the valid_token as well as each of the draft tokens in the input.
    params:
    1. num_draft_tokens: the number of draft tokens in the input
    2. num_forecast_per_token: number of forecast tokens that get associated with each draft and valid token
    3. vocab_size: the vocabulary size of the given model
    4. batch_size: this is needed to create the correct shape of the returned forecast tokens which are further appended to the input/ draft tree which is of shape [bsz, input_ids_length]
    '''
    vocab_size = vocab_size
    total_input_tokens = (1+num_draft_tokens)
    # we repeat to create the forecast tokens for valid input and each draft token
    forecast_tokens = torch.arange(vocab_size, vocab_size + num_forecast_per_token, device = device).repeat(batch_size, total_input_tokens)
    return forecast_tokens

def llm_generate_position_ids_for_draft_and_forecast_tokens(n_branch_list, pos_id_for_valid_token, num_forecast_per_token, num_draft_tokens, use_top1_expand=False, batch_size = 1, device = "cpu"):
    '''
    This API generates position IDs for both draft and forecast tokens.
    Draft tokens at the same level share the same position IDs.
    Forecast tokens get position IDs that follow the IDs of their associated valid or draft tokens.
    params:
    1. n_branch_list : the list shows the number of draft tokens at each level
    2. pos_id_for_valid_token : the position ID corresponding to the valid token in the input
    3. num_forecast_per_token : number of forecast tokens that get associated with each draft and valid token
    4. num_draft_tokens: the number of draft tokens in the input
    5. use_top1_expand: whether to use top1-expand style draft tree expansion
    '''
    pos_ids = [pos_id_for_valid_token]
    if num_draft_tokens>0:
        for n_depth in range(1, len(n_branch_list)+1):
            n_tokens_for_current_depth = int(np.prod(n_branch_list[:n_depth]))
            if use_top1_expand:
                n_tokens_for_current_depth = int(n_branch_list[n_depth-1])
            pos_ids += [pos_id_for_valid_token+n_depth] * n_tokens_for_current_depth

    forecast_pos_ids = []
    for pid in pos_ids:
        forecast_pos_ids += [pid+i for i in range(1, num_forecast_per_token + 1)]

    pos_ids += forecast_pos_ids
    return torch.tensor(pos_ids, dtype=torch.long, device = device).repeat(batch_size, 1)

def llm_build_ssd_causal_mask(input_slice, max_input_tokens, num_draft_tokens, n_branch_list,
                      num_forecast_per_token, prefix_kv_len, valid_kv_len, model_context_len, mask_neg=-100, pad_left=True, cache_index=None, use_top1_expand=False):
    '''
    This API is responsible for building the causal mask for SSD.
    It assumes that the prefix KV cache is always appended to the left of the valid KV cache.
    Additionally, the API assumes the same padding direction for both the input and the past KV.

    The valid input, draft tokens, and forecast tokens all attend to the valid past KV. Forecast tokens also attend to the prefix KV.
    
    Draft tokens at the first level attend to the valid KV/ root token in the draft tree, while tokens at subsequent levels attend to the valid token as well as their parent draft tokens.       .
    Forecast tokens follow the same routine as their corresponding draft tokens and, in addition, attend to themselves in an autoregressive manner.
    params:
    1. input_slice = refers to the current input slice without any padding
    2. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    3. num_draft_tokens: the number of draft tokens in the input
    4. n_branch_list: the list shows the number of draft tokens at each level
    5. num_forecast_per_token: number of forecast tokens that get associated with each draft and valid token
    6. prefix_kv_len: length of prefix kv cache
    7. valid_kv_len: length of valid KV cache accumulated so far
    8. model_context_len : the maximum context length that can be sent to the model (this is the HF maximum length of the context)
    9. mask_neg: mask value for the positions we don't want to attend to
    10. pad_left: whether the current input and the KV cache follow the left padding or the right padding (API assumes same for both)
    11. use_top1_expand: whether to use top1-expand style draft tree expansion
    '''
    # copied frm llama causal mask utils
    total_kv_len = model_context_len-max_input_tokens
    if pad_left:
        # Cache index should not be passed. Concat op is used in doing the KV cache update
        assert cache_index is None, "Invalid argument error: we do not support the combination of performing left padding and doing scatter for KV cache update."
    else:
        # if the user is doing right padding, it is necessary to pass the cache_index.
        assert cache_index is not None, "Invalid argument error: we do not support the combination of performing right padding and doing concat for KV cache update"
    #look at build_attention_mask_infer_bfs
    # Get batch size and seq len of input
    batch_size, input_len, = input_slice.shape[:2]
    pad_tokens = max_input_tokens - input_len
    device = input_slice.device
    dtype = input_slice.dtype

    #determine the length of padding  KV
    pad_kv_len=total_kv_len - prefix_kv_len - valid_kv_len

    #Determine the number of valid inputs (i.e exculding the draft & forecast tokens)
    num_valid_inputs = input_len - num_draft_tokens - (num_draft_tokens+1)*num_forecast_per_token
    #Create a blank causal mask that has mask_neg in all positions
    causal_mask = torch.full((batch_size, 1, max_input_tokens, total_kv_len+max_input_tokens), mask_neg, device=device, dtype=dtype)

    #Create a blank attention mask with all zeros.
    #We will set ones in the positions where we would want a token to attend
    attn_mask = torch.zeros((batch_size,1,max_input_tokens,total_kv_len+max_input_tokens),dtype=torch.int, device=device)

    #All valid tokens in input attend to valid_past_kv. Set these entries to 1.
    #Ensure that we do not set these for pad tokens
    # here we assume that the padding direction for input & the kv cache is same.
    # also we assume that the prefix kv is always append next to valid kv & towards it's left.
    if pad_left:
        attn_mask[:, :, pad_tokens:, pad_kv_len+prefix_kv_len:total_kv_len] = 1
    else:
        attn_mask[:,:,:input_len,prefix_kv_len:prefix_kv_len+valid_kv_len] = 1

    #create attention mask with lower triangle of 1 - this corresponds to the valid inputs
    valid_input_attn_mask = torch.tril(torch.ones(num_valid_inputs,num_valid_inputs),)

    #Set the valid_input_attn_mask in attn_mask.
    #The location of this would be:
    #                               Row     : 0 to  num_valid_inputs-1
    #                               Column  : total_kv_len to total_kv_len+num_valid_inputs-1
    if pad_left:
        attn_mask[:, :, pad_tokens:pad_tokens+num_valid_inputs, total_kv_len+pad_tokens:total_kv_len+pad_tokens+num_valid_inputs] = valid_input_attn_mask
    else:
        attn_mask[:,:,:num_valid_inputs, total_kv_len:total_kv_len+num_valid_inputs] = valid_input_attn_mask

    #All draft and forecast tokens should attend to valid tokens. Set those positions to 1 also
    if pad_left:
        attn_mask[:,:, pad_tokens+num_valid_inputs:max_input_tokens, total_kv_len+pad_tokens:total_kv_len+pad_tokens+num_valid_inputs] = 1
    else:
        attn_mask[:,:,num_valid_inputs:input_len, total_kv_len:total_kv_len+num_valid_inputs] = 1

    #Now we have to identify which tokens the draft tokens need to attend to
    if pad_left:
        offset_q = pad_tokens + num_valid_inputs -1
        offset_kv = total_kv_len + pad_tokens + num_valid_inputs - 1
    else:
        offset_q = num_valid_inputs - 1
        offset_kv = total_kv_len + num_valid_inputs - 1

    mask_prev_node = [[0]]
    len_verify = len(n_branch_list) if num_draft_tokens else 0

    if use_top1_expand:
        parent = mask_prev_node
        for ndx in range(len_verify):
            new_connection = [parent[0] + [int(sum(n_branch_list[:ndx])) + 1 + i] for i in range(n_branch_list[ndx])]
            mask_prev_node += new_connection
            parent = new_connection
    else:
        prev_all = [[0]]
        offset = 0
        for ndx in range(len_verify):
            curr = []
            for prev in prev_all:
                for _ in range(n_branch_list[ndx]):
                    offset += 1
                    curr += [prev + [offset]]
            prev_all = deepcopy(curr)
            mask_prev_node += prev_all


    for ndx, cmask in enumerate(mask_prev_node):
        attn_mask[:, :, offset_q + ndx, offset_kv + torch.tensor(cmask)] = 1

    # Now we identify which tokens the forecast tokens need to attend to
    mask_next_draft = []
    offset_q += len(mask_prev_node)
    for ndx_forecast, cmask in enumerate(mask_prev_node):
        offset_forecast = len(mask_prev_node) + ndx_forecast * num_forecast_per_token
        for ndx in range(num_forecast_per_token):
            mask_next_draft += [cmask + list(range(offset_forecast, offset_forecast + ndx + 1))]
    for ndx, cmask in enumerate(mask_next_draft):
        attn_mask[:, :, offset_q + ndx, offset_kv + torch.tensor(cmask)] = 1
        # prefix for next forecasting
        if pad_left:
            attn_mask[:, :, offset_q + ndx, pad_kv_len:pad_kv_len+prefix_kv_len] = 1
        else:
            attn_mask[:, :, offset_q + ndx, :prefix_kv_len] = 1

    causal_mask.masked_fill_(attn_mask.bool(), 0)
    # TODO: If pad_left is False, i.e if right padding is there, we need to adjust the mask to reflect the post KV cache update layout
    # old good KV | new KV | padding KV, the above computation assumes we have old KV + (padding) | new KV + (padding) always.
    # Causal mask is of shape [1, 1, new_KV, total_KV], we need to move around the total KV dimension.
    # Scatter the new KV cache after the cache index and make everything else, -100.
    if pad_left is False:
        cache_tensor = torch.arange(max_input_tokens).to(device = cache_index.device)
        cache_position = cache_index  + cache_tensor
        indices = cache_position.view(1, 1, 1, -1).expand(causal_mask.shape[0], causal_mask.shape[1], max_input_tokens,
                                                          cache_position.shape[-1])
        causal_mask = causal_mask.scatter(dim=-1, index=indices, src=causal_mask[:,:,:,total_kv_len:])
        causal_mask[:,:,:,prefix_kv_len+valid_kv_len+max_input_tokens:] = mask_neg
    return causal_mask

def llm_add_prefix_kv_cache(prefix_kv, current_kv, model_context_len, max_input_tokens,key_concat_axis, value_concat_axis=-2):
    '''
    This API is responsible for adding the prefix kv cache to the left of accumulated past kv (current kv to be sent into the model).
    It checks whether the addition is feasible as it should not exceed the cache budget. (model_context_len - max_input_tokens)
    # TODO: revisit to see if this budget  needs to be updated for the Long Context feature.

    params:
    1. prefix_kv: prefix KV cache for the ssd
    2. current_kv: thw current accumulated unpadded past kv cache
    3. model_context_len : the maximum context length that can be sent to the model (this is the HF maximum length of the context)
    4. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    5. key_concat_axis: the axis to which we want to append the keys
    6. value_concat_axis: the axis to which we want to concatenate the values
    '''
    #Extract the len from values
    # prefix_kv is of shape [n_layer,2, bsz, num_kv_heads, seq_len, head_dim]
    prefix_kv_len = prefix_kv[0][1].shape[-2]

    if current_kv:
        current_kv_len = current_kv[0][1].shape[-2]

        # Check if we would overflow the KV going into the model if we add prefix KV to it
        assert prefix_kv_len + current_kv_len <= model_context_len - max_input_tokens

        concatenated_key_values = tuple((_concat(prefix_key, current_key, key_concat_axis),
                                         _concat(prefix_value, current_value, value_concat_axis))
                                        for (prefix_key, prefix_value), (current_key, current_value) in
                                        zip(prefix_kv, current_kv))
        return concatenated_key_values

    return prefix_kv

def llm_verify_and_accept_draft_tokens(sample_tree, logp, n_branch_list, use_top1_expand=False) -> list:
    '''
    This function verifies and accepts draft tokens at the current step. It assumes that the token sampled from the logits of the valid token (the first token) is always accepted.
    The function then recursively checks if this accepted token matches any draft tokens at the next level, repeating the process until no match is found.

    params:
    1. sample_tree: the draft tree with the (valid token + draft tokens) + forecast tokens
    2. logp: the logits corresponding to the tokens in the sample tree
    3. n_branch_list: the list shows the number of draft tokens at each level
    4. use_top1_expand: whether to use top1-expand style draft tree expansion
    '''

    depth_max = len(n_branch_list)
    # offsets_with_children is a collection of offsets of note
    # that has children when we're doing top1-expand style tree expansion.
    # For instance, when n_branch_list=[2, 3, 2, 3] then offsets_with_children should be [1, 3, 6],
    # which can be calculated using cumulative sum of [1] + n_branch_list
    n_branch_with_root = [1] + n_branch_list
    offsets_with_children = [sum(n_branch_with_root[:idx+1]) for idx in range(len(n_branch_with_root))][:-2]
    def _verify_recursive(accepted_all, node_ids_all, accepted, node_ids, depth_current, offset):
        if depth_current <= depth_max:
            n_branch = n_branch_list[depth_current-1]
            target = accepted[0, -1]
            for branch in range(1, n_branch+1):
                ndx_node = offset + branch
                if target == sample_tree[0, ndx_node]:
                    target_next = torch.topk(logp[:, ndx_node], k=1, dim=-1).indices
                    accepted = torch.concat([accepted, target_next], dim=-1)
                    node_ids = deepcopy(node_ids) + [ndx_node]
                    accepted_all += [accepted]
                    node_ids_all += [node_ids]
                    if use_top1_expand:
                        # If current node does not have children,
                        # then we need to stop verification there.
                        # Otherwise call _verify_recursive on its children
                        if ndx_node not in offsets_with_children:
                            return accepted_all, node_ids_all
                        else:
                            offset_next = int(np.sum(n_branch_list[:depth_current]))
                    else:
                        n_branch_next = n_branch_list[depth_current] if depth_current+1 <= depth_max else 0
                        offset_next = int(np.sum(np.cumprod(n_branch_list[:depth_current]))) + (branch-1) * n_branch_next
                    accepted_all, node_ids_all = _verify_recursive(accepted_all, node_ids_all, accepted, node_ids, depth_current+1, offset_next)
        return accepted_all, node_ids_all
    target = torch.topk(logp[:, 0], k=1, dim=-1).indices
    # initial
    accepted = target
    node_ids = [0]
    depth_next = 1
    offset_next = 0
    accepted_all, node_ids_all = _verify_recursive([accepted], [node_ids], target, node_ids, depth_next, offset_next)
    return accepted_all[-1], node_ids_all[-1]


def llm_build_draft_tree_for_next_inference(valid_token, logits_of_selected_forecast_tokens, n_branch_list, use_top1_expand=False, ):
    #refer build_tree_dfs
    '''
    This function is responsible for building the draft tree for next inference using the valid token and the forecast tokens (sampled from the logits using topk)

    params:
    valid_token: previous last token
    logits_of_selected_forecast_tokens: logits outputs of the two forecast tokens of acceptted trajectory
    n_branch_list: the list shows the number of draft tokens at each level, [3,2] example , topk sampling parameter k in different levels
    use_top1_expand: whether to use top1-expand style draft tree expansion
        '''
    samples = [valid_token]
    tree_bfs = [valid_token]
    len_draft = len(n_branch_list)
    logp = torch.log_softmax(logits_of_selected_forecast_tokens, dim=-1)
    if use_top1_expand:
        for ndx in range(len_draft):
            ## remove repeated tokens, sample 1 extra token, use extra token if repeated token present
            samples = torch.topk(logp[:, ndx:ndx + 1], k=n_branch_list[ndx] + 1, dim=-1).indices
            # check if top-1 token is not repeated in next stage of draft
            last_top1_sample = tree_bfs[-1][0,0]
            idx = torch.where(last_top1_sample == samples.squeeze()[:-1])[0]
            if len(idx) > 0:
                # shift tokens
                samples = torch.cat([samples[:,:,:idx[0]],samples[:,:,idx[0]+1:]], dim=-1)
            else:
                samples = samples[:,:,:-1]

            tree_bfs += samples
    else:
        for ndx in range(len_draft):
            samples += torch.topk(logp[:, ndx:ndx + 1], k=n_branch_list[ndx], dim=-1).indices
        # refer build_tree_dfs
        for ndx in range(1, len_draft + 1):
            tree_bfs += [samples[ndx]] * int(np.prod(n_branch_list[:ndx - 1]))
    tree_bfs = torch.concat(tree_bfs, dim=1)
    return tree_bfs



def llm_get_next_step_forecast_logits(logits, last_accepted_token_index, n_branch_list, use_top1_expand=False):
    '''
    From the current ids, once we select the last draft acceptance token position, we then need to determine where do the corresponding
    forecast logits for this reside and return the forecast logits

    params:
    1. logits: the logits that correspond to the valid+draft and the associated forecast tokens
    2. last_accepted_token_index: the index of the last accepted token
    3. n_branch_list : the list shows the number of draft tokens at each level, [3,2] example , topk sampling parameter k in different levels
    '''
    len_draft = len(n_branch_list)
    offset = llm_len_flat_sample_tree(n_branch_list, use_top1_expand=use_top1_expand)
    return logits[:,offset+(last_accepted_token_index*len_draft):offset+ (last_accepted_token_index+1)*len_draft,:]

def llm_verify_ssd_arn(max_input_tokens, n_branch_list, use_top1_expand=False):
    '''
    This function determines if the ARN is sufficient to hold the valid_token+ draft_tokens + forecast tokens associated with the valid_token and each of the draft_tokens

    params:
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    n_branch_list : the list shows the number of draft tokens at each level, [3,2] example , topk sampling parameter k in different levels
    use_top1_expand: whether to use top1-expand style draft tree expansion
    '''
    len_flat_sample_tree_ = llm_len_flat_sample_tree(n_branch_list, use_top1_expand)
    draft_tree_length = len_flat_sample_tree_ + len_flat_sample_tree_*len(n_branch_list)
    return max_input_tokens>=draft_tree_length

def llm_len_flat_sample_tree(n_branch_list, use_top1_expand=False):
    '''
    This function returns the length of the flat sample tree (this only corresponds to the draft tokens length, excluding any associated forecast tokens)
    for example, for [2,3], the flat length is 2+ 2*3 = 8.=, 2 draft tokens in the first level, and 6 in the next

    params:
    1. n_branch_list : the list shows the number of draft tokens at each level, [3,2] example , topk sampling parameter k in different levels
    2. use_top1_expand: whether to use top1-expand style draft tree expansion
    '''
    if not isinstance(n_branch_list, torch.Tensor):
        n_branch_list = torch.Tensor(n_branch_list)
    if use_top1_expand:
        return 1 + int(n_branch_list.sum().item())
    return 1+int(torch.cumprod(n_branch_list, dim=0).sum().item())


def llm_preprocess_prefix_kv(prefix_kv, key_concat_axis, num_layers):
    '''
    The function is responsible for prefix keys and values in the tupled format expected by the prepared graph

    params:
    prefix_kv: the learned prefix kv with shape [28, 2, 1, 8, 16, 128], [num_layers, 2, bsz, num_kv_heads, seq_len, head_dim]
    key_concat_axis: the axis along which seq_len is present for keys
    num_layers: num_layers keys to extract
    '''
    prefix_kv_tuple = ()
    for idx in range(num_layers):
        key_states, value_states = prefix_kv[idx][0], prefix_kv[idx][1]
        if key_concat_axis==3 or key_concat_axis==-1:
            key_states = key_states.transpose(2, 3)
        prefix_kv_tuple += ((key_states, value_states),)
    return prefix_kv_tuple

def llm_check_if_end_generation(generated_tokens, tokenizer):
    '''
    The function checks if the currently generated tokens contain the eos_token.
    If it does, it sets the break_generation flag to true and extracts all tokens that appear before the eos_token from the generated tokens.

    params:
    1. generated_tokens: the tokens generated so far. [bsz, output tokens]
    2. tokenizer: the tokenizer used to decode the generated tokens
    '''
    break_generation = False
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id in generated_tokens[0]:
        stop_index = (generated_tokens[0] == eos_token_id).nonzero(as_tuple=True)
        stop_index = stop_index[0][0].item()
        break_generation = True
        return (break_generation, generated_tokens[:, :stop_index])
    return (break_generation, generated_tokens)

def llm_capture_ssd_stats(n_accepted_all, input_text, output_text, output_ids = None):
    '''
    This function is used to capture statistics like acceptance rate, input_text, etc.
    It returns a dictionary containing the computed stats as available.
    '''
    stat={}
    n_tokens = np.array(n_accepted_all) + 1
    stat['n_accepted'] = n_accepted_all
    stat['sum_n_tokens'] = int(n_tokens.sum())
    stat['avg_n_tokens'] = float(n_tokens.mean())
    stat['mem_bound_speedup'] = stat['avg_n_tokens']
    stat['input_text'] = input_text
    stat['output_text'] = output_text
    if output_ids is not None:
        stat['output_ids'] = output_ids
    return stat
