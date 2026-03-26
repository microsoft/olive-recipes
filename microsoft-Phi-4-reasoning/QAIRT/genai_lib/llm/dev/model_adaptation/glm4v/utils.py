import torch
from typing import Optional, List

try:
    from transformers_modules.modeling_glm import GlmRotaryEmbedding
except ImportError as e:
    print(f"{e} \n Please instantiate Glm4v with AutoModelForCausalLM first to have transformers_modules cache")

import functools


def lmm_update_causal_mask(prepared_1d_attention_mask, input_embeds, max_input_tokens, model_context_len, model_id_or_path, mask_neg):
    """
        Precompute causal mask
        Inputs:
            prepared_1d_attention_mask: attention mask of shape (batch_size, model_context_length)
            inputs_embeds: inputs_embeds sent to the model
            max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
            model_context_len: maximum number of tokens that the model can consume in total
            model_id_or_path: model name or path to pretrained model
            mask_neg: proxy for minus infinity since minus infinity is not quantization friendly. This value should be
                large enough to drown out tokens that should not be attended to
    """
    model = _get_model(model_id_or_path=model_id_or_path)

    cache_position = torch.arange(model_context_len-max_input_tokens, model_context_len, device = input_embeds.device)

    causal_mask = model._update_causal_mask(attention_mask=prepared_1d_attention_mask, input_tensor=input_embeds, output_attentions=True,
                              cache_position=cache_position, past_key_values=None)

    causal_mask = causal_mask.clamp_min(mask_neg)
    return causal_mask.to(dtype=input_embeds.dtype)


def llm_create_position_embeddings(config, position_ids):
    """
        Precompute rotary embeddings
        Inputs:
            config: model config
            position_ids: position_ids sent to the model
            max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    """
    assert position_ids is not None
    device = position_ids.device
    dim = int(config.head_dim * config.partial_rotary_factor)
    rotary_embed = _get_rotary_embedding(dim, config.max_position_embeddings, config.rope_theta, device)

    cos, sin = rotary_embed(torch.ones(1, device=device), position_ids.cpu())
    cos, sin = cos.unsqueeze(dim=1), sin.unsqueeze(dim=1)
    cos = cos[:, :, :, :dim // 2]
    sin = sin[:, :, :, :dim // 2]

    return (cos.to(device=device), sin.to(device=device))


@functools.cache
def _get_rotary_embedding(dim, max_position_embeddings, base, device):
    return GlmRotaryEmbedding(dim=dim, max_position_embeddings=max_position_embeddings, base=base, device=device)


def lmm_preprocess_inputs(input_ids=None, images=None, inputs_embeds=None, past_key_values=None, boi_token_id=None,
                          embedding_layer=None, vision_model=None):
    """
        Preprocess the inputs to accommodate image features to inputs_embeds, by default the first inference shouldn't
        contain past_key_values to run multimodal generation
        Inputs:
            input_ids: input_ids sent to the model
            images: images sent to the model
            inputs_embeds: embedded inputs sent to the model
            past_key_values: past_key_valyes sent to the model
            boi_token_id: begin of image token id defined in config
            embedding_layer: the embedding layer used to embed input_ids
            vision_model: the model used to generate image features from given images
        Output:
            input_embeds: embedded inputs

    """

    if past_key_values is None or len(past_key_values) == 0:
        assert inputs_embeds is None, "inputs_embeds should be generated from the input_ids and image_embeds " \
                                      "(if provided) for the first inference (none kvcache mode)"
        assert input_ids is not None, f"input_ids should be provided for the first inference"
        if not _is_empty(images):  # multi-modality
            assert embedding_layer is not None
            assert vision_model is not None

            assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
            inputs_embeds = embedding_layer(input_ids)
            multi_flags = [True if boi_token_id in input_id.tolist() else False for input_id in input_ids]

            images = images.to(dtype=inputs_embeds.dtype)
            batch_size, num_concurrent_media, num_tiles, num_channels, height, width = images.shape
            images = images.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
            images_features = vision_model(images)

            new_input_embeds = []

            """ 
                Process inputs_embeds for each sample and batch them back to tensors
                boi_token and eoi_token will be trimmed and the position id for the image features will be repeated to 
                the size of the image features
            """
            image_count = 0
            for i in range(len(input_ids)):
                input_id = input_ids[i].tolist()
                if multi_flags[i]:
                    boi_token_pos = input_id.index(boi_token_id)
                    assert boi_token_pos >= 0, "begin_of_image not found!"
                    num_image_padding_tokens = input_id.count(boi_token_id)
                    assert (
                            num_image_padding_tokens == images_features[image_count].shape[0]
                    ), f"Wrong image padding token number: {num_image_padding_tokens}"

                    new_input_embeds.append(torch.cat(
                        (inputs_embeds[i, :boi_token_pos], images_features[image_count].to(inputs_embeds.device),
                         inputs_embeds[i, boi_token_pos + num_image_padding_tokens:])))
                    image_count += 1
                else:
                    new_input_embeds.append(inputs_embeds[i])
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            return inputs_embeds

    return embedding_layer(input_ids)


def _is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if image_list is not None:
            return False
    return True


def _get_position_ids(input_ids, device):
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return position_ids


@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    from transformers_modules.modeling_glm import GlmModel
    config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
    config.num_layers = 1
    config.vision_config['num_hidden_layers'] = 1
    model = GlmModel(config)
    return model
