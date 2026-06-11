import operator
import collections
from typing import Dict, Sequence
from functools import reduce
from dataclasses import replace

import torch
from peft.tuners.lora import LoraLayer
from peft import get_peft_model, PeftModel, PeftMixedModel


def concat_adapter(
        model,
        adapters: Sequence[str],
        weights: Sequence[int],
        keep_rank: bool = True,
        fold_scaling: bool = False,
        delete_adapters: bool = True,
        concat_adapter_name: str = "concat_set"
):
    """
    Concatenate adapters in the given model. This is a variant implementation of 'cat' mode of the below API:
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/model.py
    It differs in that the ranks of concatenated adapters will be customized by each LoRA layer instead of same ranks
    across all layers.
    :param model: A model attached with LoRA adapters
    :param adapters: Adapters attached to the model
    :param weights: Weights applied to each the corresponding adapters list, must be consistent with length
    :param keep_rank: Whether to keep ranks across layers. When set to False, no zero paddings are applied
    :param fold_scaling: Whether to fold scaling into LoRA weights. When set to False, the LoRA weights will not
    be scaled by lora_scaling
    :param delete_adapters: Whether to delete adapters. When set to True, all adapters in the adapters list are deleted
    :param concat_adapter_name: Name of the concatenated adapter. Default is "concat_set"
    :return: None. Adapters are concatenated in place and concatenated adapter will be set by default
    """
    for adapter in adapters:
        if adapter not in model.peft_config.keys():
            assert ValueError(f'Cannot find adapter adapter "{adapter}" in the given model!')
    
    if keep_rank:
        if not isinstance(model, (PeftModel, PeftMixedModel)):
            # dummy_config created only to get a peft model instance
            dummy_config = replace(
                model.peft_config[adapters[0]],
                r = 1,
                lora_alpha=1,
                target_modules=[list(model.peft_config[adapters[0]].target_modules)[0]]
            )
            model = get_peft_model(model, dummy_config, adapter_name='dummy')
            model.delete_adapter('dummy')

        # PEFT API to concat adapters
        if not fold_scaling:
            original_scalings = set_scaling(model, adapters, 1.0)
        model.add_weighted_adapter(
            adapters=adapters,
            weights=weights,
            adapter_name=concat_adapter_name,
            combination_type='cat'
        )

        # Set lora scaling for targeted adapters' LoRA layers
        for layer_name, target in model.named_modules():
            if isinstance(target, LoraLayer):
                # NOTE: there might be risk of directly writing into lora_scaling
                loras_scaling = []

                for adapter in adapters:
                    if adapter in target.lora_A:
                        current_scaling = torch.full((target.r[adapter],), original_scalings[adapter][layer_name])
                    elif adapter in target.lora_embedding_A:
                        raise RuntimeError(
                            f'We only support LoRA weights for now but LoRA embeddings are found in {adapter} instead!'
                        )
                    else:
                        continue

                    loras_scaling.append(current_scaling)

                if len(loras_scaling) == 0:
                    raise ValueError(
                        f'No matching LoRAs found for {layer_name}, please check your adapter configuration!')

                target.scaling[concat_adapter_name] = torch.zeros(
                    model.peft_config[concat_adapter_name].r, dtype=target.weight.dtype, device=target.weight.device
                )
                loras_scaling = torch.cat(loras_scaling)
                target.scaling[concat_adapter_name][: loras_scaling.shape[0]].copy_(loras_scaling)
    else:
        concat_rank = sum([model.peft_config[adapter].r for adapter in adapters])
        concat_modules = reduce(operator.or_, [model.peft_config[adapter].target_modules for adapter in adapters])
        # We use the config to initialize an adapter but ranks/scalings will change for some target_modules
        # So this config is never meant to be used or accessed external to this function
        concat_config = replace(
            model.peft_config[adapters[0]],
            r=concat_rank,
            lora_alpha=concat_rank,
            target_modules=concat_modules
        )

        model = get_peft_model(model, concat_config, adapter_name=concat_adapter_name)

        for layer_name, target in model.named_modules():
            if isinstance(target, LoraLayer):
                if concat_adapter_name in target.lora_A:
                    concat_lora_A = target.lora_A[concat_adapter_name]
                    concat_lora_B = target.lora_B[concat_adapter_name]
                elif concat_adapter_name in target.lora_embedding_A:
                    raise RuntimeError(
                        f'We only support LoRA weights for now but LoRA embeddings are found in {concat_adapter_name} instead!'
                    )
                else:
                    continue

                # NOTE: there might be risk of directly writing into lora_scaling
                loras_A, loras_B, loras_scaling = [], [], []
                for adapter in adapters:
                    if adapter in target.lora_A:
                        current_adapter_lora_A = target.lora_A[adapter].weight
                        current_adapter_lora_B = target.lora_B[adapter].weight
                    elif adapter in target.lora_embedding_A:
                        raise RuntimeError(
                            f'We only support LoRA weights for now but LoRA embeddings are found in {adapter} instead!'
                        )
                    else:
                        continue

                    loras_A.append(current_adapter_lora_A.data)
                    loras_B.append(current_adapter_lora_B.data)
                    loras_scaling.append(torch.full((target.r[adapter], ), target.scaling[adapter]))

                if len(loras_A) == 0:
                    raise ValueError(f'No matching LoRAs found for {layer_name}, please check your adapter configuration!')

                loras_A = torch.cat(loras_A, dim=0)
                loras_B = torch.cat(loras_B, dim=1)

                # This might be risky since we're replacing the whole parameter
                if isinstance(concat_lora_A, torch.nn.Conv2d):
                    target.lora_A[concat_adapter_name] = torch.nn.Conv2d(
                        loras_A.shape[1],
                        loras_A.shape[0],
                        kernel_size=concat_lora_A.kernel_size,
                        stride=concat_lora_A.stride,
                        padding=concat_lora_A.padding,
                        bias=concat_lora_A.bias is not None,
                        device=concat_lora_A.weight.device,
                        dtype=concat_lora_A.weight.dtype,
                    )
                    target.lora_B[concat_adapter_name] = torch.nn.Conv2d(
                        loras_B.shape[1],
                        loras_B.shape[0],
                        kernel_size=concat_lora_B.kernel_size,
                        stride=concat_lora_B.stride,
                        padding=concat_lora_B.padding,
                        bias=concat_lora_B.bias is not None,
                        device=concat_lora_B.weight.device,
                        dtype=concat_lora_B.weight.dtype,
                    )
                elif isinstance(concat_lora_A, torch.nn.Linear):
                    target.lora_A[concat_adapter_name] = torch.nn.Linear(
                        loras_A.shape[1],
                        loras_A.shape[0],
                        bias=concat_lora_A.bias is not None,
                        device=concat_lora_A.weight.device,
                        dtype=concat_lora_A.weight.dtype,
                    )
                    target.lora_B[concat_adapter_name] = torch.nn.Linear(
                        loras_B.shape[1],
                        loras_B.shape[0],
                        bias=concat_lora_B.bias is not None,
                        device=concat_lora_B.weight.device,
                        dtype=concat_lora_B.weight.dtype,
                    )

                target.lora_A[concat_adapter_name].weight.data.copy_(loras_A)
                target.lora_B[concat_adapter_name].weight.data.copy_(loras_B)
                target.scaling[concat_adapter_name] = torch.cat(loras_scaling).to(model.device)

    model.set_adapter(concat_adapter_name)
    if delete_adapters:
        for adapter in adapters:
            model.delete_adapter(adapter)

def set_scaling(
        model,
        adapters: Sequence[str],
        scaling: float = 1.
) -> Dict[str, Dict[str, float]]:
    """
    Set scaling of LoRA layers for adapters to scaling
    :param model: The LoRA model to be rescaled
    :param adapters: Adapters of which to rescale
    :param scaling: The scaling factor to be applied to all target adapters
    :return: The original scaling for each adapter's target modules
    """
    original_scalings = collections.defaultdict(dict)

    for layer_name, target in model.named_modules():
        if isinstance(target, LoraLayer):
            # NOTE: there might be risk of directly writing into lora_scaling
            for adapter in adapters:
                if adapter in target.scaling:
                    original_scalings[adapter][layer_name] = target.scaling[adapter]
                    target.scaling[adapter] = scaling
                elif adapter in target.lora_embedding_A:
                    raise RuntimeError(
                        f'We only support LoRA weights for now but LoRA embeddings are found in {adapter} instead!'
                    )
                else:
                    continue

    for adapter in adapters:
        if not original_scalings[adapter]:
            raise RuntimeError(f'No matching LoRAs found for {adapter}, please check your adapter configuration!')

    return original_scalings
