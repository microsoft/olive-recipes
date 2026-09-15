#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""  This file provides utilities that the pipeline would need to work with the adaptations made to LLaMa model. """

import torch
import functools
from aimet_torch.utils import place_model, change_tensor_device_placement
from transformers.models.mllama import modeling_mllama
from transformers.cache_utils import DynamicCache
from genai_lib.llm.evaluation_utils import llm_compute_loss_from_logits
from genai_lib.llm.dev.model_adaptation.mllama.adaptation import QcMllamaForConditionalGeneration
from tqdm import tqdm


def llm_update_causal_mask(prepared_1d_attn_mask, input_tensor, max_input_tokens, model_context_len, model_id_or_path, mask_neg = -100.0, cache_index=None, pad_to_left = True):
    '''
    This function creates a causal mask (2D) from the 1D attention mask

    params:
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor : input_ids/ input_embeddings
    3. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    4. model_context_len: maximum number of tokens that the model can consume in total
    5. model_id: Model name or path to pretrained model
    6. mask_neg: proxy for minus infinity since minus infinity is not quantization friendly. This value should be large
    enough to drown out tokens that should not be attended to
    7. cache_index: the index for the starting position of kvcaches
    6. pad_to_left: determines if the KV cache is padded to the left or right
    '''
    llama_model = _get_model(model_id_or_path)

    # if the cache position is None, then we assume that the current input ids will be concatenated to the right end and
    # #hence we construct the cache position accordingly to be sent into the update_causal_mask

    if pad_to_left:
        # Cache index should not be passed. Concat op is used in doing the KV cache update
        assert cache_index is None, "Invalid argument error: we do not support the combination of performing left padding and doing scatter for KV cache update."
    else:
        # if the user is doing right padding, it is necessary to pass the cache_index.
        assert cache_index is not None, "Invalid argument error: we do not support the combination of performing right padding and doing concat for KV cache update"

    if cache_index is None:
        cache_position = torch.arange(model_context_len-max_input_tokens, model_context_len, device = input_tensor.device)
    else:
        cache_position = torch.arange(max_input_tokens, dtype=torch.float32, device=input_tensor.device) + cache_index.to(input_tensor.device)

    input_embeds = torch.ones((input_tensor.shape[0], input_tensor.shape[1], 1), device = input_tensor.device)
    prepared_attention_mask = llama_model._update_causal_mask(attention_mask=prepared_1d_attn_mask, input_tensor=input_embeds, output_attentions=True,
                              cache_position=cache_position, past_key_values=None)
    prepared_attention_mask = prepared_attention_mask.clamp_min(mask_neg)
    return prepared_attention_mask


def llm_create_position_embeddings(config, position_ids=None):
    '''
    This function creates position embedding (RoPE) from the position ids.
    params:
    1. config: model configuration to create the LLamaRotaryEmbedding object, expect config for backward compatibility, in future transformers, we only expect to pass the config, the one we have in docker, takes in the req argument
    2. position_ids: required position ids passed into the model
    '''

    hidden_size = config.hidden_size
    max_position_embeddings = config.text_config.max_position_embeddings
    num_attention_heads = config.text_config.num_attention_heads
    rope_theta = config.text_config.rope_theta
    dim = int((hidden_size // num_attention_heads))
    device = position_ids.device
    x = torch.ones(1, device=device)
    rotary_emb = _get_rotary_embedding(config=config.text_config, device=device)
    cos, sin = rotary_emb(x, position_ids=position_ids)
    cos, sin = cos.unsqueeze(dim = 1), sin.unsqueeze(dim = 1)
    cos = cos[:,:,:,:dim//2]
    sin = sin[:,:,:, :dim//2]
    return cos, sin


def _get_rotary_embedding(config, device):
    from transformers.models.mllama.modeling_mllama import MllamaRotaryEmbedding
    rotary_emb = MllamaRotaryEmbedding(config=config, device=device)
    return rotary_emb


@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    from transformers.models.mllama.modeling_mllama import MllamaTextModel
    config = AutoConfig.from_pretrained(model_id_or_path)
    config.text_config.num_hidden_layers = 1
    model = MllamaTextModel(config.text_config)
    return model


@torch.no_grad()
def create_cross_attention_masks(input_ids, image_token, vision_context_length, mask_value=-100.0, device="cpu", useful_vision_cache_size=None):
    """
    Creates cross-attention and MLP masks for a multimodal transformer model that processes both text and vision inputs.

    This function generates two types of masks:
    1. `cross_attention_mask`: Used in cross-attention layers to mask out vision tokens before a specific image token appears in the input.
    2. `full_text_row_masked_out_mask`: Used in MLP layers to mask out text tokens before the image token.

    The masks are constructed based on the position of a special `image_token` in the `input_ids`. Tokens before this position are masked differently
    for attention and MLP operations.

    Args:
        input_ids (torch.Tensor): Tensor of shape (N,) or (1, N) containing token IDs, including a special image token.
        image_token (int): The token ID that marks the start of vision-related tokens.
        vision_context_length (int): The number of vision tokens to attend over.
        mask_value (float, optional): The value used to mask out tokens in the attention mask. Default is -100.0.
        device (str, optional): The device on which tensors are allocated. Default is "cpu".
        useful_vision_cache_size (int, optional): If set, limits the number of vision tokens considered useful for attention.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - cross_attention_mask (torch.Tensor): Mask of shape (1, 1, N, vision_context_length) for cross-attention layers.
            - full_text_row_masked_out_mask (torch.Tensor): Mask of shape (1, 1, N, 1) for MLP layers.

    Raises:
        Exception: If `image_token` is not found in `input_ids`.
    """

    # Cross-Attn does mask add, hence we want mask until image token, and zeros afterwards
    attn_array = torch.ones(input_ids.shape[-1], dtype=torch.int, device=device) * mask_value
    # MLP does mask multiply, hence we want zeros until image token, and ones afterwards
    mlp_array = torch.zeros(input_ids.shape[-1], dtype=torch.int, device=device)

    # Find the position of the first occurrence of target
    pos = (input_ids.flatten() == image_token).nonzero(as_tuple=True)[0]
    if pos.numel() > 0:
        pos = pos[0].item()
        attn_array[pos + 1:] = 0
        mlp_array[pos + 1:] = 1
    else:
        raise Exception(f"image_token not present in input_ids provided")

    # Reshape and expand the binary array to create the masks
    cross_attention_mask = attn_array.reshape(1, input_ids.shape[-1], 1).expand(1, 1, input_ids.shape[-1], vision_context_length).clone() # (1, 1, ARN, visionCL)
    full_text_row_masked_out_mask = mlp_array.reshape(1, 1, input_ids.shape[-1], 1).clone() # (1, 1, ARN, 1)

    if useful_vision_cache_size and useful_vision_cache_size < vision_context_length:
        # Mask all image tokens beyond the useful_vision_cache_size
        cross_attention_mask[:, :, :, useful_vision_cache_size:] = mask_value

    return cross_attention_mask, full_text_row_masked_out_mask


@torch.no_grad()
def generate_vision_cache(lmm_model, pixel_values, tokens_per_frame, vision_context_length, head_size, vision_encoder=None, device="cpu", return_dict=True, include_self_attn_cache=True):
    """
    Generates a vision key-value (KV) cache for use in a multimodal language model (LMM) that integrates visual and textual inputs.

    This function prepares the vision cache by either:
    - Using the model's built-in vision encoder and projector to compute cross-attention key/value states, or
    - Accepting an external `vision_encoder` that directly outputs the vision cache.

    The cache is structured to support attention mechanisms in the language model, with optional support for self-attention cache initialization.

    Args:
        lmm_model (PreTrainedModel): The multimodal language model containing vision and language components.
        pixel_values (Union[torch.Tensor, List[torch.Tensor]]): Image tensor(s) to be encoded into vision features.
        tokens_per_frame (int): Number of tokens allocated per image frame in the vision context.
        vision_context_length (int): Total number of vision tokens in the context.
        head_size (int): Dimensionality of each attention head.
        vision_encoder (Optional[Callable], optional): External encoder that outputs vision KV cache. If None, uses the model's internal encoder. Default is None.
        device (str, optional): Device to perform computation on (e.g., "cpu" or "cuda"). Default is "cpu".
        return_dict (bool, optional): If True, returns a `DynamicCache` object; otherwise, returns a legacy cache format. Default is True.
        include_self_attn_cache (bool, optional): Whether to initialize self-attention cache placeholders. Default is True.

    Returns:
        Union[DynamicCache, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
            - If `return_dict` is True, returns a `DynamicCache` object with `key_cache` and `value_cache`.
            - Otherwise, returns a legacy tuple format of (key_cache, value_cache).

    Raises:
        AssertionError: If `include_self_attn_cache` is False and multiple images are passed.
        AssertionError: If the external `vision_encoder` does not return the expected number of layers.
    """

    if isinstance(pixel_values, torch.Tensor):
        pixel_values = [pixel_values]

    if not include_self_attn_cache:
        assert len(pixel_values) == 1, f"When simulating on-device VisionEncoder behavior, we should pass a single image at a time"

    # Initialze Cache with placeholders
    key_cache, value_cache = list(), list()
    if include_self_attn_cache:
        for block_idx in range(lmm_model.config.text_config.num_hidden_layers):
            if block_idx in lmm_model.config.text_config.cross_attention_layers:
                multi_frame_key_states = torch.zeros((1, lmm_model.config.text_config.num_attention_heads, vision_context_length, head_size), device=device).clone()
                multi_frame_value_states = torch.zeros((1, lmm_model.config.text_config.num_attention_heads, vision_context_length, head_size), device=device).clone()
                if getattr(lmm_model.config.text_config, "transposed_key_cache", False):
                    multi_frame_key_states = multi_frame_key_states.transpose(2, 3)
                key_cache.append(multi_frame_key_states)
                value_cache.append(multi_frame_value_states)
            else:
                empty_tensor = torch.tensor((), device=device, dtype=next(lmm_model.language_model.parameters()).dtype)
                empty_key = empty_tensor.reshape(1, lmm_model.config.text_config.num_key_value_heads, -1, head_size)
                empty_value = empty_tensor.reshape(1, lmm_model.config.text_config.num_key_value_heads, -1, head_size)
                if getattr(lmm_model.config.text_config, "transposed_key_cache", False):
                    empty_key = empty_key.transpose(2, 3)
                key_cache.append(empty_key)
                value_cache.append(empty_value)

    # Loop through images and scatter image embeddings onto vision KV Cache
    for img_idx, img_pixels in enumerate(pixel_values):

        if vision_encoder is None: # Use original model for cache generation if vision_encoder not provided
            with place_model(lmm_model.vision_model, device):
                *_, C, H, W = img_pixels.shape
                img_pixels = img_pixels.reshape(-1, C, H, W) # adapted model expects 4D = [B, C, H, W]
                list_image_features = lmm_model.vision_model.forward_multiscales(img_pixels.to(device))

            with place_model(lmm_model.multi_modal_projector_2, device):
                cross_attention_states = list_image_features[getattr(lmm_model, "select_vision_scale", -1)]
                cross_attention_states = lmm_model.multi_modal_projector_2(cross_attention_states).reshape(-1, cross_attention_states.shape[-2], lmm_model.hidden_size)

            with place_model(lmm_model.language_model, device):
                # Generate cache
                for block_idx in lmm_model.config.text_config.cross_attention_layers:
                    if block_idx >= lmm_model.config.text_config.num_hidden_layers:
                        break
                    key_states, value_states = lmm_model.language_model.model.layers[block_idx].cross_attn.compute_vision_kv_cache(cross_attention_states)
                    if include_self_attn_cache:
                        if getattr(lmm_model.config.text_config, "transposed_key_cache", False):
                            key_cache[block_idx][:,:,:,tokens_per_frame*img_idx:tokens_per_frame*(img_idx+1)] = key_states
                        else:
                            key_cache[block_idx][:,:,tokens_per_frame*img_idx:tokens_per_frame*(img_idx+1),:] = key_states
                        value_cache[block_idx][:,:,tokens_per_frame*img_idx:tokens_per_frame*(img_idx+1),:] = value_states
                    else: # When preparing standalone Vision Encoder, we don't want scatter op as part of graph, it should be handled externally by an orchest
                        key_cache.append(key_states)
                        value_cache.append(value_states)

        else: # Generate vision cache with vision_encoder
            with place_model(vision_encoder, device):
                *_, C, H, W = img_pixels.shape
                img_pixels = img_pixels.reshape(-1, C, H, W) # adapted model expects 4D = [B, C, H, W]
                vision_cache = vision_encoder(img_pixels)
                assert len(lmm_model.config.text_config.cross_attention_layers) == len(vision_cache), f"vision_encoder should output cache for all cross-attn layers"
                for block_idx, block_cache in zip(lmm_model.config.text_config.cross_attention_layers, vision_cache):
                    key_states = block_cache[0]
                    value_states = block_cache[1]
                    if getattr(lmm_model.config.text_config, "transposed_key_cache", False):
                        key_cache[block_idx][:,:,:,tokens_per_frame*img_idx:tokens_per_frame*(img_idx+1)] = key_states.transpose(2,3)
                    else:
                        key_cache[block_idx][:,:,tokens_per_frame*img_idx:tokens_per_frame*(img_idx+1),:] = key_states
                    value_cache[block_idx][:,:,tokens_per_frame*img_idx:tokens_per_frame*(img_idx+1),:] = value_states

    cache = DynamicCache()
    cache.key_cache = key_cache
    cache.value_cache = value_cache

    if not return_dict:
        return cache.to_legacy_cache()

    return cache


@torch.no_grad()
def mllama_evaluate_ppl(lmm_model, dataloader, processor, vision_context_length, tokens_per_frame, head_size, vision_encoder=None, num_batches=50):
    """
    Evaluates the perplexity (PPL) of a multimodal LLaMA-based language model (LMM) on a dataset containing both text and image inputs.

    This function iterates over a specified number of batches from a dataloader, computes the negative log-likelihood (NLL) for each batch,
    and returns the exponential of the average NLL as the perplexity score. It supports both standard and vision-augmented LLaMA models,
    including those with external vision encoders.

    Args:
        lmm_model (PreTrainedModel): The multimodal LLaMA model to evaluate. Can be a standard or vision-augmented variant.
        dataloader (DataLoader): A PyTorch DataLoader yielding batches of multimodal data with keys like 'input_ids', 'attention_mask', and 'pixel_values'.
        processor (Callable): A processor or tokenizer used to encode and decode text and special tokens (e.g., image tokens).
        vision_context_length (int): Total number of vision tokens in the context window.
        tokens_per_frame (int): Number of vision tokens allocated per image frame.
        head_size (int): Dimensionality of each attention head in the model.
        vision_encoder (Optional[Callable], optional): External vision encoder that outputs vision key-value caches. If None, the model's internal encoder is used.
        num_batches (int, optional): Number of batches to evaluate. Defaults to 50.

    Returns:
        float: The computed perplexity score over the evaluated batches.

    Notes:
        - This function disables gradient computation for efficiency.
        - It supports both models that compute vision key-value caches internally and those that rely on an external vision encoder.
        - The function assumes that the model returns a loss-compatible output when called with the appropriate inputs.
    """

    num_batches = num_batches if num_batches else len(dataloader)
    nlls = []
    device = next(lmm_model.language_model.parameters()).device

    for batch_id, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Evaluating Perplexity")):
        if batch_id >= num_batches:
            break
        batch = change_tensor_device_placement(batch, device)

        # Adapted model (QcMllamaForConditionalGeneration) that computes visionKV as part of Vision Encoder
        if isinstance(lmm_model, QcMllamaForConditionalGeneration):
            vision_cache = generate_vision_cache(lmm_model=lmm_model, pixel_values=batch['pixel_values'], tokens_per_frame=tokens_per_frame, vision_context_length=vision_context_length,
                                                 head_size=head_size, vision_encoder=vision_encoder, device=device, return_dict=True, include_self_attn_cache=True)
            num_frames = batch['pixel_values'].shape[0]
            useful_vision_cache_size = tokens_per_frame * num_frames
            cross_attention_mask, full_text_row_masked_out_mask = create_cross_attention_masks(batch['input_ids'],
                                                                                               image_token = processor.tokenizer.encode(processor.image_token)[-1],
                                                                                               vision_context_length = vision_context_length,
                                                                                               useful_vision_cache_size = useful_vision_cache_size,
                                                                                               device=device)
            outputs = lmm_model.language_model(input_ids = batch['input_ids'],
                            attention_mask = batch['attention_mask'],
                            cross_attention_mask = cross_attention_mask,
                            full_text_row_masked_out_mask = full_text_row_masked_out_mask,
                            past_key_values = vision_cache)

        # Original model that computes visionKV as part of LLM
        else:
            outputs = lmm_model.language_model(input_ids = batch['input_ids'],
                                               attention_mask = batch['attention_mask'],
                                               pixel_values = batch['pixel_values'].unsqueeze(0).unsqueeze(0),
                                               aspect_ratio_ids = torch.tensor(1, dtype=torch.int64, device=device).reshape(1, 1),
                                               aspect_ratio_mask = torch.tensor(1, dtype=torch.int64, device=device).reshape(1, 1, 1))

        nlls.append(llm_compute_loss_from_logits(outputs, batch["input_ids"]))
        del outputs, batch
    ppl = torch.exp(torch.stack(nlls).mean())
    return float(ppl)
