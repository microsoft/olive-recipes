#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Tuple
import torch

def _set_tree_mask(model, tree_mask):
    model.tree_mask = tree_mask

def llm_verify_draft_tree(
        target_model: torch.nn.Module,
        draft_tokens: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        retrieve_indices: torch.Tensor,
        target_past_key_values: Tuple[Tuple[torch.Tensor]],
        prev_input_len: int):
    """
    Verify draft tree using target model
    Inputs:
        target_model: target model for verification
        draft_tokens: draft tokens to verify
        tree_mask: attention mask for draft tree
        tree_position_ids: position id for draft tree
        retrieve_indices: indices of candidate tokens of draft tree paths
        target_past_key_values: target model KV cache
        prev_input_len: length of previous input tokens
    Output:
        accepted_tokens: accepted tokens verified by target model
        accepted_hidden_states: accepted hidden states verified by target model
        accepted_logits: accepted logits verified by target model with shape (#accepted_tokens, #vocab)
        accepted_token_indices: accepted token indices
        target_past_key_values: updated target model KV cache
    """

    _set_tree_mask(target_model, tree_mask)

    tree_position_ids = tree_position_ids[None] + prev_input_len

    tree_logits, target_past_key_values, (hidden_state,) = target_model(
        draft_tokens,
        past_key_values=target_past_key_values,
        position_ids=tree_position_ids,
        output_hidden_states=True,
    )
    if isinstance(hidden_state, (tuple, list)):
        last_hidden_state = hidden_state[-1]
    else:
        last_hidden_state = hidden_state

    tree_logits = tree_logits[0, retrieve_indices]
    padding = torch.full((1, 1), -1, dtype=torch.long, device=draft_tokens.device)
    padded_draft_tokens = torch.cat((draft_tokens, padding), dim=1)
    draft_tokens_2d = padded_draft_tokens[0, retrieve_indices]

    # Find the tokens that match the maximum logits for each position in the sequence
    posterior_mask = (
        draft_tokens_2d[:, 1:].to(tree_logits.device) == torch.argmax(tree_logits[:, :-1], dim=-1)
    ).int()

    # posterior_mask value is 1 when a token in a tree path is accepted or not
    # so we need to apply cumprod to mask out the latter tokens after the first rejected tokens
    # e.g., [1, 1, 0, 1, 0, 0] -> [1, 1, 0, 0, 0, 0]
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()

    # Choose the best candidate
    if accept_length == 0:
        # Default to the first candidate if none are accepted
        best_candidate = torch.tensor(0, dtype=torch.long, device=draft_tokens_2d.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

    # Postprocessing
    accepted_token_indices = retrieve_indices[best_candidate, :accept_length + 1]
    accepted_hidden_states = last_hidden_state[:, accepted_token_indices]
    accepted_logits = tree_logits[best_candidate, :accept_length + 1]
    accepted_tokens = draft_tokens_2d[best_candidate, :accept_length + 1]

    _set_tree_mask(target_model, None)
    return accepted_tokens, accepted_hidden_states, accepted_logits, accepted_token_indices, target_past_key_values, 


def llm_rerank_and_prune_draft_tree(
    root_token, draft_tokens, scores_list, parents_list, total_tokens
):
    """
    Rerank and prune the draft tree.

    Input:
    root_token: root of draft tree. concated before pruned draft_tokens.
    draft_tokens: list of draft tokens for each depth.
    scores_list: list of scores for each depth.
    parents_list: list of parents for each depth.
    total_tokens: total tokens to be kept after pruning draft tree.

    Output:
    pruned_tokens: pruned tokens after reranking and pruning.
    shape=(1, total_tokens+1)
    [[root_token_id, ...`totla_tokens` draft tokens after pruning]]

    retrieve_indices: list of indices for reconstructing path to leaves.
                      The indices after the leaf are indicated as -1.
    shape=(#leaf, max_depth)
    [
        [0, ...indices of the path to leaf node in the tree, -1, ..., -1],
        ...size of leaves in the pruned tree
    ]

    tree_mask: causal mask for pruned_tokens.
                1 for attendable, 0 for not attendable. The tokens on the path to current token are attendable.
    shape=(1, 1, total_tokens+1, total_tokens+1)
    [
        [1, 1, 0, 0, ..., 0]
        [1, 0, 1, 0, ..., 0]
        ...
        [1, 0, 0, 1, ..., 1]
    ]

    tree_position: position ids of each token in the pruned tree.
                   Tokens at same depth have the same position id.
    shape=(total_tokens+1,)
    [0, 1, 1, 1, ..., max_depth]
    """

    #
    # Referred topK_generate of Eaglet reference code
    #

    top_k = draft_tokens[0].shape[-1]
    flat_scores_list = torch.cat(scores_list, dim=0).view(-1)
    flat_draft_tokens = torch.cat(draft_tokens, dim=0).view(-1)

    top_scores = torch.topk(flat_scores_list, total_tokens, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values

    # Pick `total_tokens` tokens with highest scores
    pruned_tokens = flat_draft_tokens[top_scores_index]
    pruned_tokens = pruned_tokens[None]
    # Prepend `root_token` to pruned draft tree
    pruned_tokens = torch.cat((root_token, pruned_tokens), dim=1)

    # `draft_parents` is the parent of each token in `pruned_tokens`
    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
    # `mask_index` is the index of mask for parent
    mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)

    # Update the index of `mask_index` considering the prepended root
    mask_index[draft_parents == 0] = -1
    mask_index = mask_index + 1
    mask_index_list = mask_index.tolist()

    tree_mask = torch.eye(total_tokens + 1).bool()
    # The root token is visible to all draft tokens
    tree_mask[:, 0] = True
    # Update `tree_mask` by accumulating the mask of parent
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

    tree_position_ids = torch.sum(tree_mask, dim=1) - 1

    tree_mask = tree_mask.float()[None, None]

    max_depth = torch.max(tree_position_ids) + 1
    # `mask_index` has the index of parents.
    noleaf_index = torch.unique(mask_index).tolist()
    # Decrease 1 due to the prepended root token.
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num

    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
    retrieve_indices = retrieve_indices.tolist()

    rid = 0
    position_ids_list = tree_position_ids.tolist()

    # Contruct `retrieve_indices` for reconstructing path to leaves
    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = position_ids_list[i]
            # Update path by walking from the leaf to the root.
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    return pruned_tokens, retrieve_indices, tree_mask, tree_position_ids


def llm_create_draft_tree(
    draft_model,
    draft_token_logits,
    hidden_state,
    past_key_values,
    depth,
    top_k,
    tokens_seen_so_far,
    forward_kwargs,
    trimmed_vocab_map=None
):
    """
    Create the draft tree with draft model.

    Input:
    draft_model: model to generate the draft tokens to be added to the draft tree.
    draft_token_logits: logits for constructing first depth of draft tree. logits are come from root token.
    hidden_state: hidden state of root token from draft model.
    past_key_values: KV cache for draft model
    depth: depth of draft tree
    top_k: how many draft tokens are picked for draft tree
    tokens_seen_so_far: the length added to position ids.
    forward_kwargs: other arguments for draft model forward function.
    trimmed_vocab_map: vocabulary map to get original token index from trimmed token index

    Output:
    draft_tokens: tuple of draft tokens for each depth.
    [
      [...`top_k` tokens ids at 1st depth],
      [...`top_k`**2 tokens ids at 2nd depth],
      ...,
      [...`top_k`**2 tokens ids at {`depth` + 1}-th depth]
    ]

    scores_list: tuple of scores for each depth.
    [
        [...scores for `top_k` tokens at 1st depth],
        [...cumulative scores for `top_k`**2 tokens at 2nd depth],
        ...,
        [...cumulative scores for `top_k`**2 tokens at {`depth` + 1}-th depth]
    ]

    parents_list: tuple of parents for each depth.
    [
        [0], # the parent index for tokens at 1st depth which indicates the root token.
        [...indices for `top_k` tokens at 2nd depth],
        ...,
        [...indices for `top_k` tokens at {`depth` + 1}-th depth],
    ]
    """

    #
    # Referred topK_generate of Eaglet reference code
    #
    draft_tokens = []
    scores_list = []
    parents_list = []

    device = hidden_state.device

    # Pick `top_k` from `draft_token_logits` to construct first depth of draft tree.
    last_p = torch.log_softmax(draft_token_logits[:, -1], dim=-1)
    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values

    if trimmed_vocab_map is not None:
        topk_index = trimmed_vocab_map[topk_index]

    scores = topk_p[0]
    # Store scores of the draft tokens for cummulative scores.
    scores_list.append(scores[None])
    # Initialize the parent index of the first depth as the root token.
    parents_list.append(torch.zeros(1, dtype=torch.long, device=device))
    # Initialize draft tree with `top_k` tokens in 1st depth.
    draft_tokens.append(topk_index)

    # Prepare inputs for forward of draft model.
    input_ids = topk_index
    input_hidden = hidden_state.repeat(1, top_k, 1)
    tree_mask_init = torch.eye(top_k, device=device)[None, None]
    tree_mask = tree_mask_init
    cum_topk_index = torch.arange(top_k, device=device)
    position_ids_init = torch.zeros(top_k, dtype=torch.long, device=device)
    len_posi = tokens_seen_so_far

    for i in range(depth):
        _set_tree_mask(draft_model, tree_mask)
        position_ids = len_posi + position_ids_init
        position_ids = position_ids.expand((input_ids.shape[0], -1))

        (last_headout, past_key_values, out_hidden) = draft_model(
            hidden_states=input_hidden,
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            output_hidden_states=True,
            **forward_kwargs,
        )

        # Update position offset for next inference
        len_posi += 1

        # Update `parents_list` considering depth with `bias`
        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        # The flattened draft tree looks like [root_token, ...`top_k` tokens at 1st depth, ...`top_k`**2 tokens at n-th depth, ...]
        # `bias` indicates the number of tokens in the previous depth:
        # - `1` for the root token
        # - `bias1` for `top_k` tokens at 1st depth.
        # - `bias2` for `top_k`**2 tokens at n-th depth (n > 1).
        bias = 1 + top_k**2 * bias2 + bias1
        # `cum_topk_index` indicates the offset of `top_k` tokens in the current depth
        parents = cum_topk_index + bias
        parents_list.append(parents)

        # Applied log scale to use addition when accumulating probabilities
        # and to prevent the cumulative magnitude from becoming too small.
        last_p = torch.log_softmax(last_headout[0], dim=-1)

        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        if trimmed_vocab_map is not None:
            topk_index = trimmed_vocab_map[topk_index]

        # Update cumulative scores of the last `top_k`**2 draft tokens.
        cum_scores = topk_p + scores[:, None]

        # Pick `top_k` tokens from the last `top_k`**2 draft tokens.
        cum_topk = torch.topk(cum_scores.view(-1), top_k, dim=-1)
        cum_topk_index, cum_topk_p = cum_topk.indices, cum_topk.values
        scores = cum_topk_p

        # Pick hidden state of parents of top_k tokens for next inference.
        # By dividing by `top_k`, we can get the indices of parent among the input tokens.
        out_ids = cum_topk_index // top_k
        input_hidden = out_hidden[-1][:, out_ids]

        input_ids = topk_index.view(-1)[cum_topk_index][None]

        # Update `draft_tokens` with all `top_k` tokens per each draft token.
        draft_tokens.append(topk_index)
        scores_list.append(cum_scores)
        tree_mask = torch.cat((tree_mask[:, :, out_ids], tree_mask_init), dim=3)

    _set_tree_mask(draft_model, None)

    return tuple(draft_tokens), tuple(scores_list), tuple(parents_list)


def llm_apply_tree_mask(
    causal_mask, tree_mask, max_input_tokens, mask_neg=-100.0, pad_to_left=True, cache_index=None
):
    """
    Mask out causal mask wiht tree mask.

    Inputs:
        causal_mask: causal mask with shape (batch_size, 1, input_length, context_length)
        tree_mask: mask consisting of 0 and 1 with shape (input_length, tree_size)
        max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
        mask_neg: proxy for minus infinity since minus infinity is not quantization friendly.
                  This value should be large enough to drown out tokens that should not be attended to
        pad_to_left: boolean value indicating whether padding is done towards the left or right.
        cache_index: cache_index determines where the attn_mask_input was placed.
                     If None, the input_attention mask was placed towards the end.

    Outputs:
        causal_mask: masked out `causal_mask` with `tree_mask`

    Example: max_input_tokens = 4
    Left padding case)
        cache_index = 6
        causak_mask: shape=(1,1,4,10) like [2 paddings, 4 valid kv$, 1 padding, 3 valid inputs]
        [[[
            [mask_neg, mask_neg, 0,          0,          0,          0, mask_neg,   mask_neg,   mask_neg, mask_neg],
            [mask_neg, mask_neg, 0,          0,          0,          0, mask_neg,          0,   mask_neg, mask_neg],
            [mask_neg, mask_neg, 0,          0,          0,          0, mask_neg,          0,          0, mask_neg],
            [mask_neg, mask_neg, 0,          0,          0,          0, mask_neg,          0,          0,        0],
        ]]]

        tree_mask: shape=(1,1,3,6)
        [[[
            [1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
        ]]]

        causal_mask after applying tree_mask: *mask_neg*s are masked out by tree_mask
        [[[
            [mask_neg, mask_neg, 0,          0,          0,          0, mask_neg,   mask_neg,   mask_neg, mask_neg],
            [mask_neg, mask_neg, 0,          0, *mask_neg*, *mask_neg*, mask_neg,          0,   mask_neg, mask_neg],
            [mask_neg, mask_neg, 0,          0, *mask_neg*,          0, mask_neg, *mask_neg*,          0, mask_neg],
            [mask_neg, mask_neg, 0, *mask_neg*,          0, *mask_neg*, mask_neg, *mask_neg*, *mask_neg*,        0],
        ]]]

    Right padding case)
        cache_index = 4
        causak_mask: shape=(1,1,4,10) like [4 valid kv$, 3 valid inputs, 3 paddings]
        [[[
            [0,          0,          0,          0,          0,   mask_neg, mask_neg, mask_neg, mask_neg, mask_neg],
            [0,          0,          0,          0,          0,          0, mask_neg, mask_neg, mask_neg, mask_neg],
            [0,          0,          0,          0,          0,          0,        0, mask_neg, mask_neg, mask_neg],
            [0,          0,          0,          0,          0,          0,        0, mask_neg, mask_neg, mask_neg],
        ]]]

        tree_mask: shape=(1,1,3,6)
        [[[
            [1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
        ]]]

        causal_mask after applying tree_mask: *mask_neg*s are masked out by tree_mask
        [[[
            [0,          0, *mask_neg*, *mask_neg*,          0,   mask_neg, mask_neg, mask_neg, mask_neg, mask_neg],
            [0,          0, *mask_neg*,          0, *mask_neg*,          0, mask_neg, mask_neg, mask_neg, mask_neg],
            [0, *mask_neg*,          0, *mask_neg*, *mask_neg*, *mask_neg*,        0, mask_neg, mask_neg, mask_neg],
            [0,          0,          0,          0,          0,          0,        0, mask_neg, mask_neg, mask_neg],
        ]]]

    """
    _, _, input_length, tree_mask_target_len = tree_mask.shape
    remained_target_length = tree_mask_target_len - input_length

    cache_index = causal_mask.shape[-1] - max_input_tokens if cache_index is None else cache_index
    if pad_to_left:
        causal_mask[:, :, -input_length:, cache_index+max_input_tokens-input_length:cache_index+max_input_tokens][
            tree_mask[:, :, :, -input_length:] == 0
        ] = mask_neg
        causal_mask[:, :, -input_length:, cache_index - remained_target_length : cache_index][
            tree_mask[:, :, :, :-input_length] == 0
        ] = mask_neg
    else:
        causal_mask[:, :, :input_length, cache_index-remained_target_length:cache_index+input_length][
            tree_mask == 0
        ] = mask_neg

    return causal_mask


def llm_evaluate_block_efficiency(output_ids, accept_lengths, stop_token_ids=None):
    """
    Evaluate block eficiency about output before the stope sequence.

    Inputs:
        output_ids: the generated ids for evaluation Block Efficiency.
        accept_lengths: list of lengths of accepted tokens for each iteration.
        stop_sequences: stop sequences used to stop generation.

    Outputs:
        block_efficiency: average accepted length for each iteration.
        generation_length: length of the output excluding stop sequences.
        output_ids: output_ids before the stop sequence.
    """
    iter_count = len(accept_lengths)
    if stop_token_ids is not None:
        if not isinstance(stop_token_ids, (tuple, list)):
            stop_token_ids = [stop_token_ids]
        stop_indices = [i for i, id in enumerate(output_ids) if id in stop_token_ids]
        if len(stop_indices) > 0:
            stop_index = stop_indices[0]
            if accept_lengths[-1] == 1:
                # Ignore last iteration if the last accepted token is solely stop token.
                iter_count = iter_count - 1

            output_ids = output_ids[:stop_index]

    block_efficiency = len(output_ids) / (iter_count)
    generation_length = len(output_ids)

    return block_efficiency, generation_length, output_ids