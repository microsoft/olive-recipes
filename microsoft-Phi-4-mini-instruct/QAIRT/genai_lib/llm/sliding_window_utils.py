#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch

def create_swa_mask_from_global_mask(global_mask, sliding_window_length, lookback_window, ssd_prefix_kv_len, global_cache_index=None):
    """
    This API creates the swa_mask from the global causal mask. We expect the global_cache_index and sliding_cache_index are aligned (not in values, but they should progress in the same manner) for API correctness.
    params:
    global_mask: this is the global causal mask of the shape [1, 1, ARN, context_length].
                We fetch a subset of values from this to create the sliding causal mask of shape [1, 1, ARN, sliding_window_length]
    sliding_window_length: this is the expanded sliding window shape [could be inclusive of prefix KV$ or not, could be +1/-1, or even have more than ARN]->
                            all we want to check if the global_cache_index + ARN fall within or outside the sliding_window_length
    lookback_window: Consider this as the lookback_window, every token in our mask should ideally be looking at exactly lookback_window worth of actual tokens
    global_cache_index: this integer tells the amount of past KV$ [the new_kv$ will be added right after the old KV$, starting from global_cache_index position]
    ssd_prefix_kv_len: the length of forecast prefix KV$

    Note: we only support the right padding scenario for SWA+SSD computation in simulation
    """

    max_input_tokens = global_mask.shape[2]
    if global_cache_index!=None:
        # Right padding case, if ssd_prefix_kv_len == 0 then it is non-ssd right padding use-case.

        swa_causal_mask = torch.ones((1, 1, max_input_tokens, sliding_window_length-ssd_prefix_kv_len)).to(global_mask.device)*global_mask.min().item()

        # Step 1:
        TARGET_VALUE = 0.0
        # We know the boundary is around global_cache_index..

        # max is because when you have a case when the updated sliding window is more than where your cache index is, then we don't want to simply return the first updated sliding window values from the global mask because we want to make sure that the actual tokens in that are still attending to lookback_window tokens.
        left_end = max(0, global_cache_index + max_input_tokens - sliding_window_length)
        right_end = left_end + sliding_window_length
        # Shift left_end due to prefix_kv_len
        left_end += ssd_prefix_kv_len

        #  The following loop ensures that for every element in ARN, we need to ensure that as we travel back, we pick lookback amount of indices..
        for i in range(max_input_tokens):
            # get the indices which equal to zero first, we only want the lookback_window worth of indices
            mask = (global_mask[:,:,i, left_end:right_end]==TARGET_VALUE).int()

            # Compute prefix sum from right to left
            cumulative_sum = torch.flip(torch.cumsum(torch.flip(mask, dims=[-1]), dim=-1), dims=[-1])

            # Keep only positions where prefix sum ≤ max_matches
            final_mask = (mask.bool()) & (cumulative_sum <= lookback_window)


            # Use final_mask to copy values from global_mask to swa_causal_mask
            # We need to broadcast final_mask to match the shape of global_mask[:, :, i, left_end:right_end]
            selected_values = torch.where(final_mask, global_mask[:, :, i, left_end:right_end], torch.tensor(global_mask.min().item(), device=global_mask.device))

            # Copy into the correct slice of swa_causal_mask
            swa_causal_mask[:, :, i, :] = selected_values

        # Step 2: We concat the extracted causal mask with the prefix KV$ portion on the left
        swa_causal_mask = torch.cat((global_mask[:,:,:,:ssd_prefix_kv_len], swa_causal_mask.to(global_mask.device)), dim=-1)
        return swa_causal_mask

    else:
        # For left padding, we do not support SSD for now.
        swa_causal_mask = torch.ones((1, 1, max_input_tokens, sliding_window_length-ssd_prefix_kv_len)).to(global_mask.device)*global_mask.min().item()

        # Step 1:
        #the TARGET_VALUE represents the value in the causal_mask where the tokens attend to
        TARGET_VALUE = 0.0
        left_end = global_mask.shape[-1] - sliding_window_length + ssd_prefix_kv_len
        right_end = global_mask.shape[-1]

        # The for loop ensures that for every element in ARN, we need to ensure that as we travel back, we pick lookback amount of indices..
        for i in range(max_input_tokens):
            # get the indices which equal to zero first, we only want the lookback_window worth of indices
            mask = (global_mask[:,:,i, left_end:right_end]==TARGET_VALUE).int()

            # Compute prefix sum from right to left
            cumulative_sum = torch.flip(torch.cumsum(torch.flip(mask, dims=[-1]), dim=-1), dims=[-1])

            # Keep only positions where prefix sum ≤ max_matches
            final_mask = (mask.bool()) & (cumulative_sum <= lookback_window)

            # Use final_mask to copy values from global_mask to swa_causal_mask
            # We need to broadcast final_mask to match the shape of global_mask[:, :, i, left_end:right_end]
            selected_values = torch.where(final_mask, global_mask[:, :, i, left_end:right_end], torch.tensor(global_mask.min().item(), device=global_mask.device))

            # Copy into the correct slice of swa_causal_mask
            swa_causal_mask[:, :, i, :] = selected_values

        # Step 2: We concat the extracted causal mask with the prefix KV$ portion on the left
        swa_causal_mask = torch.cat((global_mask[:,:,:,:ssd_prefix_kv_len], swa_causal_mask.to(global_mask.device)), dim=-1)

        return swa_causal_mask