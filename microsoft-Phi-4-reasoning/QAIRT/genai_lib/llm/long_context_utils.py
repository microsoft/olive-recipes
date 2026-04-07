#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""
Utility APIs for supporting Key Similarity Long Context and Sliding Window
"""
# as per design the most updated design considering pytorch & multihead all bmm.
import os
import pickle
import torch
import torch.nn as nn
import aimet_torch.nn.modules.custom as modules
from genai_lib.llm.test_vectors import to_torch_tensor,to_cpu
from typing import Tuple, List, Optional

MatMul = modules.MatMul
class AnchorUpdaterKeySecond(nn.Module):
    def __init__(self, alpha: float = 0.001):
        super().__init__()
        self.alpha = alpha
        self.one_minus_alpha = 1 - alpha
        self.mul = MatMul()

    def forward(
        self,
        new_keys: torch.Tensor,
        valid_token_mask: torch.Tensor,
        old_anchor: torch.Tensor,
    ):
        """
        inputs:
            new_keys: [bsz, heads, ar_n, head_dim]
            valid_token_mask: [bsz, heads, 1, ar_n]
            old_anchor: [bsz, heads, 1, head_dim]
        outputs:
            new_anchor: [bsz, heads, 1, head_dim]
        """
        new_anchor = self.mul(
            valid_token_mask,  # [bsz, heads, 1, ar_n]
            new_keys,  # [bsz, heads, ar_n, head_dim]
        )  # [bsz, heads, 1, head_dim]
        new_anchor = new_anchor + self.one_minus_alpha * old_anchor
        return new_anchor


class ScorerKeySecond(nn.Module):

    def __init__(self, num_keys):
        super().__init__()
        self.mul = nn.ModuleList([MatMul() for _ in range(num_keys)])

    def forward(self, keys: tuple, anchor_buffer: tuple):
        """
        inputs:
            keys: tuple of length config.num_hidden_layers
                  where each item is of shape [bsz, heads, head_dim, context_len]

            anchor: tuple of length config.num_hidden_layers
                    where each item is of shape [bsz, heads, 1, head_dim]
        outputs:
            score:  tuple of length config.num_hidden_layers
                    where each item is of shape [bsz, heads, 1, contex_len]
        """
        score = ()
        for i in range(len(keys)):
            # anchor[i] shape   [bsz, heads, 1, head_dim]
            # keys[i] shape     [bsz, heads, head_dim, context_len]
            # curr_score shape  [bsz, heads, 1, context_len]
            curr_score = self.mul[i](anchor_buffer[i], keys[i])
            score += (curr_score,)

        return score


def get_scorer_input_output_names(num_hidden_layers):

    """
    This function is responsible for returning a list of the model input and output names based on the number of hidden layers for scoring network
    num_hidden_layers: number of hidden layers of the languge model

    :params
    1. num_hidden_layers : num_hidden_layers in language model config

    """
    def _get_names(pfx, sfx, n_layers):
        all = []
        for i in range(n_layers):
            all.append(f'{pfx}_{i}_{sfx}')
        return all

    input_names=[]
    input_names += _get_names("keys", "in", num_hidden_layers)
    input_names += _get_names("anchor_buffer", "in", num_hidden_layers)

    output_names=[]
    output_names += _get_names("score", "out", num_hidden_layers)
    return input_names, output_names




def generate_scorer_test_vectors(fp_language_model, fp_scoring_model, quantsim, num_hidden_layers, input_ids, batch_index, output_dir):
    """
    This function is responsible for generating test vectors for fp,qt model in output_dir
    :params
    1. fp_language_model: fp adapted language model
    2. fp_scoring_model: fp scoring network model
    3. scorer quantsim model : quantsim of scoring network model
    4. num_hidden_layers: num_hidden_layers of language model
    5. input_ids: input_ids inputs for language model
    6. batch_index: batch index of dataloader
    7. output_dir: output directory for test vector generation
    """
    test_vectors_dir = os.path.join(output_dir,'test_vectors')
    os.makedirs(test_vectors_dir, exist_ok=True)
    fp_vectors_pickle_dict = {}
    qt_vectors_pickle_dict = {}
    language_model_outputs = fp_language_model(input_ids=input_ids)
    model_input = {}
    model_input["keys"] = ()
    for n_layer in range(num_hidden_layers):
        keys = language_model_outputs[1][n_layer][0]
        model_input["keys"] += (keys,)

    model_input["anchor_buffer"] = fp_language_model.anchor_buffer
    fp_scoring_model_out = fp_scoring_model(model_input["keys"], model_input["anchor_buffer"])
    fp_vectors_pickle_dict[str(batch_index)]= model_input
    fp_vectors_pickle_dict[str(batch_index)]["score"] = fp_scoring_model_out

    qsim_scoring_model_out = quantsim.model(model_input["keys"], model_input["anchor_buffer"])
    qt_vectors_pickle_dict[str(batch_index)] = model_input
    qt_vectors_pickle_dict[str(batch_index)]["score"] = qsim_scoring_model_out
    with open(os.path.join(test_vectors_dir, f"fp_{batch_index}.pkl"), "wb") as fp_pickle_file:
        pickle.dump(to_cpu(to_torch_tensor(fp_vectors_pickle_dict)), fp_pickle_file)
    with open(os.path.join(test_vectors_dir, f"qt_{batch_index}.pkl"), "wb") as qt_pickle_file:
        pickle.dump(to_cpu(to_torch_tensor(qt_vectors_pickle_dict)), qt_pickle_file)


def llm_compute_scores(scorer, past_key_values, anchor, valid_kv_len=None, pad_to_left=True):
    """
    The function calculates the scores, where a higher score indicates greater similarity between the anchor vector and the given key
    It calls the scorer for each layer and returns a tuple of scores in the format (1, n_heads, 1, ctx_len)*n_layers (assuming keys are second input to matmul).

    We consider two scenarios for the past_key_values:

    1. Static Shape Input in the Scorer: The past_key_values contain padded key-value pairs, assuming the scorer requires a static shape input.
    The valid_kv_length is used to identify true non-padded key-value pairs. To ensure padding values are always evicted, we update the scores by
    maximizing/ inflating the scores corresponding to padding.

    2. Dynamic Shape Input in the Scorer: The scorer can accept dynamic shape input, in which case the past_key_values are not padded.

    :params
    1. scorer: a callable which computes scores given the keys and the anchor vector
    2. past_key_values: past_key_values for all the layers
    3. anchor: the anchor vector tuple representing the anchor vector for each layer, this is essentially the exponential moving average of the keys seen so far
    4. valid_kv_len: valid kv length represents the amount of valid kv in the past_key_values in case of padding
    5. pad_to_left: boolean value indicating whether padding is done towards the left or right

    """
    updated_scores = ()
    all_keys = tuple(k for k,_ in past_key_values)
    scores = scorer(all_keys, anchor)

    for score in scores:
        if valid_kv_len is not None:
            max_values, _ = torch.max(score, dim=3, keepdim=True)

            if pad_to_left:
                score[:, :, : ,:-valid_kv_len] = max_values
            else:
                score[:, :, :, valid_kv_len:] = max_values

        updated_scores += (score,)

    # the assertion ensures that there is parity between the shape of scores and past_kv shape along the sequence dimension.
    # dim represents the dimension along the sequence_length
    assert updated_scores[0].shape[-1] == past_key_values[0][1].shape[2]
    return updated_scores


def llm_compute_indices_to_keep(scores, model_context_len, max_input_tokens,
                              new_KV_per_inference, eviction_frequency = 1, num_kv_to_evict=None):
    '''
    The function is responsible for computing the indices to either keep or evict based on the scores.
    If the user does not specify how many key-value pairs (kv), we determine it dynamically using the maximum allowed kv cache budget,
    the eviction frequency, and the eviction amount at each frequency (which varies between the prefill and inference stages).

    params:

    1. scores: Tensor representing the similarity of the keys with the anchor.
    2. model_context_len : the maximum context length that can be sent to the model (this is the HF maximum length of the context)
    3. max_input_tokens: the maximum tokens that can be sent to the model, in our context represents the AR length
    4. eviction_frequency: The eviction frequency refers to how often (after how many inferences) this API is called.
        If the frequency is 1, it implies that this API is called after every inference.
        If the frequency is more than 1, it implies that eviction will be called once every "frequency" inferences
        in which case we should ensure that we create enough room to accommodate new KV until the next opportunity
        to evict.
    5. new_KV_per_inference: This is the number of tokens whose KV we expect will get added at each inference. Ideally,
        for the prefill stage, we set it to ARN, and 1 for inference (for SSD, this is the forecast tokens).
    6. num_kv_to_evict: Optionally, the user can specify the exact amount of kv to evict. If this is provided, we do
        not compute the num_kv_to_evict. If it is not provided we assume that the new_KV that was output by the language
        model is not in the candidate list for eviction.
    '''
    #score_per_layer = (bsz, num_kv_heads, 1, seq_len)
    current_len = scores[0].shape[-1]  # 4051 if dynamic tensor/ 4096 if static

    if num_kv_to_evict is None:
        # Reduce the budget by the sum of max_input_tokens and new_KV_per_inference.
        # This is because we have new KV from the latest inference that is pending addition.
        # In addition we need to create room for the next max_input_tokens that will be sent in during next inf.
        budget = model_context_len - (max_input_tokens + new_KV_per_inference)

        num_kv_to_evict = current_len - budget
        num_kv_to_evict += (eviction_frequency - 1) * new_KV_per_inference

    keep_indices_tuple = ()
    num_kv_to_keep = current_len - num_kv_to_evict
    assert num_kv_to_keep > 0, "The num_kv_to_evict should not exceed the current kv length"
    for score in scores:
        # in order to run topk, we flip the scores by multiplying with a -1 so we get the actual lowest scores
        keep_indices = torch.topk(-1 * score, k=num_kv_to_keep, dim=-1)[1]
        # needed to move the indices in the third dimension (aligned with shape of value cache) where seq_len is in third dimension
        keep_indices = keep_indices.transpose(-1, -2)
        keep_indices_tuple += (keep_indices,)
    return keep_indices_tuple


def reindexer_default(keep_indices, past_key_values):
    reindexer = Reindexer(keep_indices, past_key_values[1].shape[-1])

    updated_key = reindexer.reindex_tensor(past_key_values[0].transpose(-1, -2)).transpose(-1, -2)
    updated_value = reindexer.reindex_tensor(past_key_values[1])

    return (updated_key, updated_value)


def llm_compress_kv_cache(past_key_values, keep_indices, compression_algorithm=reindexer_default):
    '''
    The function is responsible for compressing the kv cache whenever the past_kv exceeds the budget.
    It uses the gather op internally.

    :params
    1. past_key_values: past_key_values to be compressed
    2. keep_indices: tensor representing the keep indices of shape [bsz, num_heads, indices, 1]
    3. compression_algorithm: the caller takes in the keep_indices and the kv for a given layer and returns the compressed kv.

    '''
    max_kv_idx = past_key_values[0][1].shape[2]-1

    # following check ensures that the keep_indices don't exceed the kv length.
    for layer_idx in range(len(past_key_values)):
        # (bsz, num_heads, keep_indices, 1) -> (num_heads)
        max_indices_per_head =  torch.max(keep_indices[layer_idx], dim =2)[0].squeeze(dim=0).squeeze(dim=-1)
        assert torch.all(max_indices_per_head <= max_kv_idx), "keep_indices should fall within the input past_key_values range"

    updated_past_key_values = ()
    for layer_idx in range(len(past_key_values)):
        updated_key_value = compression_algorithm(keep_indices[layer_idx], past_key_values[layer_idx], )
        updated_past_key_values += (updated_key_value,)

    return updated_past_key_values

class Reindexer:
    """
    Reindexer takes in the tensor and the keep_indices and performs gather to retain the values corresponding to keep_indices.
    """
    def __init__(self, keep_indices: torch.Tensor, head_dim: int):
        # keep_idx.shape = [bsz, heads, window, 1]
        self.keep_idx = keep_indices.sort(-2).values

        # idx.shape =  [bsz, heads, window, dim]
        self.idx = self.keep_idx.expand(-1, -1, -1, head_dim)

    def reindex_tensor(self, states_tensor: torch.Tensor, sequence_dim=-2) -> torch.Tensor:
        return torch.gather(states_tensor, sequence_dim, self.idx)

def llm_update_overwriting_cache(model, scores, num_extra_kvs):
    '''
    The API is responsible for adding the necessary indices to evict from the scores into the overwriting_index_cache. We first determine if we need to add indices to the overwriting_index_cache.

    1. If this is the initial instance of populating the cache, we add the indices to the overwriting_index_cache.

    2. If the remaining indices in the cache are fewer than num_extra_kvs, we retrieve the indices from the scores. Note that these scores do not take in the  the new Key Cache in their computation.

    We iterate over the layers and fill the overwriting_index_cache.

    params:
    model: The object that holds the overwriting index cache and the configuration head dimension.
    scores: A tuple of scores for each layer, with the shape [batch_size, num_heads, 1, window].
    num_extra_kvs: This value indicates the extent to which the concatenation of accumulated and new key-value pairs exceeds the budget. It represents the number of indices required from the overwriting_index_cache.
    '''

    if not hasattr(model, 'overwriting_index_cache'):
        raise AttributeError("The passed model does not support lazy eviction")
    overwriting_index_cache = model.overwriting_index_cache.get(0, None)

    # the flag to check whether we need to write the indices into the overwriting_index_cache
    overwrite_cache = False
    if overwriting_index_cache is None:
        overwrite_cache = True
    else:
        if overwriting_index_cache.shape[-2] < num_extra_kvs:
            overwrite_cache = True
    if overwrite_cache:
        for layer_idx in range(len(scores)):
            head_dim = model.config.head_dim

            # this was taken from the systems implementation, I believe they want to rule out any scenarios where the model.overwriting_index_len (perhaps set incorrectly by the user) is less than num_extra_kvs
            k = max(model.overwriting_index_len, num_extra_kvs)

            #(bsz, n_heads, 1, past_kv_len) -> (bsz, n_heads, 1, k) -> (bsz, n_head, k, 1)
            top_k_idx = torch.topk(scores[layer_idx], k, dim=-1)[1].transpose(-1, -2)

            # Expand indices in the last dim to match head_dim
            #(bsz, n_head, seq_len, 1) -> (bsz, n_head, seq_len, head_dim) similar to value cache shape
            model.overwriting_index_cache[layer_idx] = top_k_idx.expand(
                -1, -1, -1, head_dim
            )

def llm_scatter_exceeded_kv_using_lazy_eviction(model, past_key_values, num_extra_kvs, key_concat_axis,
                                                             value_concat_axis,):
    '''
    This API is responsible for scattering the exceeded KV cache into the indices of the old KV that will undergo eviction.
    params:
    model: this object contains the overwriting_index_cache
    past_key_values: the accumulated old and new KV cache
    num_extra_kvs: This value indicates the extent to which the concatenation of accumulated and new key-value pairs exceeds the budget. It represents the number of indices required from the overwriting_index_cache.
    key_concat_axis: the axis to which we want to append the keys
    value_concat_axis: the axis to which we want to append the values

    '''
    if not hasattr(model, 'overwriting_index_cache'):
        raise AttributeError("The passed model does not support lazy eviction")

    # container to store the updated KV cache
    updated_past_key_values = ()
    total_num_kvs = past_key_values[0][1].shape[2]
    for layer_idx in range(len(past_key_values)):
        # extract the portion of KV which equals the budget size
        cached_key = past_key_values[layer_idx][0][..., :total_num_kvs-num_extra_kvs]
        cached_value =past_key_values[layer_idx][1][..., :total_num_kvs-num_extra_kvs, :]

        # extract the exceeded_KV portion
        exceed_key = past_key_values[layer_idx][0][..., total_num_kvs-num_extra_kvs:]
        exceed_value =past_key_values[layer_idx][1][..., total_num_kvs-num_extra_kvs:, :]

        # Extract the top / front most num_extra_kvs indices from the overwriting_index_cache
        #idx (bsz, n_head, window, head_dim) -> (bsz, n_head, num_extra_kvs, head_dim)
        scatter_idx = model.overwriting_index_cache[layer_idx][..., :num_extra_kvs, :]


        # Scattering new exceeded values into old value cache at scatter index positions
        # the shape of scatter indices align with the value cache dimesnion, seq_len in the dim=-2
        updated_value = torch.scatter(cached_value, value_concat_axis, scatter_idx, exceed_value)

        # Scattering new exceeded keys into old key cache at scatter index positions
        # transpose scatter_idx if keys are transposed
        updated_key = torch.scatter(cached_key, key_concat_axis, scatter_idx.transpose(-1, -2) if key_concat_axis==3 else scatter_idx, exceed_key)


        # update the overwriting_index_cache to remove the top num_extra_kvs values
        model.overwriting_index_cache[layer_idx] = model.overwriting_index_cache[layer_idx][
                                                  ..., num_extra_kvs:, :
                                                  ]

        updated_past_key_values += ((updated_key, updated_value),)

    return updated_past_key_values

def replenish_rotating_index_cache(
    cache_length: int,
    num_kv_heads: int,
    head_dim: int,
    bsz: int = 1,
    rotating_eviction_cache: Optional[torch.Tensor] = None,
    num_sinked_kvs: int = 0,
) -> torch.Tensor:
    """
    Infer missing indices and replenish them at the end.
    params:
    cache_length: this represents the size of the queue in window dimension, for sliding window this is equal to sliding_window_context_len - ARN
    num_kv_heads: num_kv_heads in the model config
    head_dim: head_dim in the model_config.
    bsz: batch size, default is 1
    rotating_eviction_cache: the queue of shape [bsz, num_kv_heads, window, head_dim]
    num_sinked_kvs: number of sinked kvs, default is 0

    returns:
    torch.Tensor: The replenished rotating eviction cache.
        - If num_sinked_kvs > 0, the returned cache shape is [bsz, num_kv_heads, cache_length - num_sinked_kvs, head_dim].
        - Otherwise, the shape is [bsz, num_kv_heads, cache_length, head_dim].
    """

    # Full range of possible indices
    full_indices = torch.arange(cache_length)

    # Find missing (evicted) indices
    mask = torch.ones(cache_length, dtype=torch.bool)

    if rotating_eviction_cache is not None:
        # Filter out indices that are outside the current cache length
        rotating_eviction_cache = rotating_eviction_cache[:, :, rotating_eviction_cache[0, 0, :, 0] < cache_length, :]
        mask[rotating_eviction_cache[0,0,:,0]] = False
    if num_sinked_kvs > 0:
        # If there are sinked kvs, we need to remove them from the mask
        # This makes the rotating index cache shorter by num_sinked_kvs
        mask[:num_sinked_kvs] = False
    evicted = full_indices[mask]
    evicted = evicted[None, None, :, None]
    evicted = evicted.expand(bsz, num_kv_heads, -1, head_dim)

    # Append them back
    if rotating_eviction_cache is not None:
        rotating_eviction_cache = torch.cat([rotating_eviction_cache, evicted], dim=2)
    else:
        rotating_eviction_cache = evicted
    return rotating_eviction_cache


def llm_scatter_exceeded_kv_using_rotating_eviction(
    rotating_eviction_cache: torch.Tensor,
    past_key_values: Tuple[Tuple[torch.Tensor]],
    num_extra_kvs: int,
    key_concat_axis: int,
    value_concat_axis: int,
    layer_indices_to_perform_eviction: Optional[List[int]] = None,
) -> Tuple[Tuple[Tuple[torch.Tensor]], torch.Tensor]:
    """
    This API is responsible for scattering the exceeded KV cache into the indices of the old KV that will undergo eviction.
    params:
    rotating_eviction_cache: the queue of shape [bsz, num_kv_heads, window, head_dim]
    past_key_values: the accumulated old and new KV cache
    num_extra_kvs: This value indicates the extent to which the concatenation of accumulated and new key-value pairs exceeds the budget. It represents the number of indices required from the overwriting_index_cache.
    key_concat_axis: the axis to which we want to append the keys
    value_concat_axis: the axis to which we want to append the values
    layer_idx_to_perform_eviction: a list indicating which layers should undergo eviction, the ones not in the list are kept as is in the returned KV$. If None, all layers are assumed to undergo eviction.

    """
    if layer_indices_to_perform_eviction is None:
        # If no layers are specified, we assume all layers should undergo eviction
        layer_indices_to_perform_eviction = list(range(len(past_key_values)))

    assert len(layer_indices_to_perform_eviction) > 0, "At least one layer should be specified for eviction"
    assert num_extra_kvs <= rotating_eviction_cache.shape[-2], "The number of extra KVs exceeds the size of the rotating eviction cache"

    # container to store the updated KV cache
    updated_past_key_values = []
    total_num_kvs = past_key_values[layer_indices_to_perform_eviction[0]][1].shape[2]
    assert total_num_kvs >= num_extra_kvs, "The total number of KVs should be greater than or equal to the number of extra KVs"

    # Extract the top / front most num_extra_kvs indices from the rotating_eviction_cache
    # idx (bsz, n_head, window, head_dim) -> (bsz, n_head, num_extra_kvs, head_dim)
    scatter_idx = rotating_eviction_cache[..., :num_extra_kvs, :]
    scatter_idx = scatter_idx.to(past_key_values[0][0].device)
    for layer_idx, (key_cache, value_cache) in enumerate(past_key_values):
        if layer_idx in layer_indices_to_perform_eviction:
            # extract the portion of KV which equals the budget size
            cached_key = key_cache[..., :total_num_kvs - num_extra_kvs]
            cached_value = value_cache[..., :total_num_kvs - num_extra_kvs, :]

            # extract the exceeded_KV portion
            exceed_key = key_cache[..., total_num_kvs - num_extra_kvs:]
            exceed_value = value_cache[..., total_num_kvs - num_extra_kvs:, :]

            # Scattering new exceeded values into old value cache at scatter index positions
            # the shape of scatter indices align with the value cache dimension, seq_len in the dim=-2
            updated_value = torch.scatter(cached_value, value_concat_axis, scatter_idx, exceed_value)

            # Scattering new exceeded keys into old key cache at scatter index positions
            # transpose scatter_idx if keys are transposed
            updated_key = torch.scatter(cached_key, key_concat_axis,
                                        scatter_idx.transpose(-1, -2) if key_concat_axis == 3 else scatter_idx,
                                        exceed_key)

        else:
            updated_key = key_cache
            updated_value = value_cache

        updated_past_key_values += ((updated_key, updated_value),)

    # update the rotating_eviction_cache to remove the top num_extra_kvs values

    rotating_eviction_cache = rotating_eviction_cache[..., num_extra_kvs:, :]

    # Add the scatter_idx back into the rotating_eviction_cache in the end to preserve the temporal ordering of indices.
    rotating_eviction_cache = torch.cat([rotating_eviction_cache, scatter_idx.to(rotating_eviction_cache.device)], dim=2)

    return tuple(updated_past_key_values), rotating_eviction_cache


def llm_mask_lower_diagonal_swa_attention(swa_attention_mask: torch.Tensor,
                                          sliding_window:int,
                                          sliding_kv_length: int,
                                          prefix_kv_len: int = 0,
                                          mask_neg: int = -100.0,
                                          pad_to_left: bool = True) -> torch.Tensor:
    """
    This API is responsible for masking out the lower triangular area of the SWA attention mask

    params:
    swa_attention_mask: The input sliding window attention mask to be modified
    sliding_window: The window size of the sliding window attention
    sliding_kv_length: the length of kvcache for the sliding window attention
    mask_neg: mask value for the positions we don't want to attend to

    """
    # Get ARN from swa_attention_mask
    max_input_tokens = swa_attention_mask.shape[-2]

    # Lower diagonal masking offset
    diagonal_offset = -1 if pad_to_left else (sliding_kv_length - sliding_window - 1 + max_input_tokens)
    diagonal_offset += prefix_kv_len

    # Mask out lower diagonal part of SWA attention to simulate HF SWA behavior
    left_masking_area = torch.tril(
        torch.ones_like(swa_attention_mask, dtype=bool),
        diagonal=diagonal_offset
    )

    # Don't mask prefix KV
    left_masking_area[..., :prefix_kv_len] = False

    # Set the mask value
    swa_attention_mask.masked_fill_(left_masking_area, mask_neg)
    return swa_attention_mask
