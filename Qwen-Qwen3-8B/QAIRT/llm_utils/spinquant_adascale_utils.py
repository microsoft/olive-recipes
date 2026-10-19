#!/usr/bin/env python3
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""
SpinQuant + AdaScale utilities for the weight-quantization notebook.

SpinQuant R1 helpers for rotating LoRA adapter weights:
- `apply_spinquant_r1_to_adapter` applies Hadamard-based, 1/sqrt(dim)-normalized rotations to
  LoRA adapter weights (LHS rotation on `lora_A`, RHS rotation on `lora_B`). Layers are selected
  by name substrings; for VLMs separate hidden sizes / filters can be given for the language and
  vision submodules.
- `capture_norm_fusion_factors` / `apply_norm_scaling_to_lora` capture and replay the RMSNorm
  fold onto LoRA adapters.

AdaScale / quantsim helpers (used in section 4 of the notebook):
- `ONNXExportableModuleWithCache` wraps a text decoder + LM-head so AIMET can JIT-trace / build a
  QuantizationSimModel over ONLY the language path.
- `_set_blocks` swaps (QDQ) decoder / lm_head blocks back into the full HF model for PEFT eval /
  export.

Prompt-masking loss helpers (AdaScale on VLMs):
- `get_assistant_header_token_ids` / `find_response_start_index` locate the prompt/response
  boundary in a tokenized sample so the AdaScale loss can score response tokens only.
"""

from __future__ import annotations
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import torch
import copy
from transformers import DynamicCache
from aimet_torch.experimental.spinquant.hadamard_utils import get_hadamard_matrix

import logging
# Self-contained logger for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.WARN)


def apply_spinquant_r1_to_adapter(
    peft_state_dict,
    language_hidden_size = None,
    vision_hidden_size = None,
    language_layers_filter = "lang",  # "" if LLM, "lang" if VLM
    vision_layers_filter = "vis",
    layers_with_lhs_rotations = ("q_proj", "k_proj", "v_proj", "qkv", "gate_proj", "up_proj"),
    layers_with_rhs_rotations = ("o_proj", "down_proj"),
):
    """Apply SpinQuant R1 (Hadamard) rotations to LoRA adapter weights in a PEFT state dict.

    This function performs:
      * LHS rotation for `lora_A` weights of selected layers:
        `A' = (R^T @ A^T)^T`
      * RHS rotation for `lora_B` weights of selected layers:
        `B' = (B^T @ R)^T`

    The rotation matrix `R` is constructed as a Hadamard matrix normalized by `1/sqrt(dim)`,
    where `dim` is the hidden size of the respective submodule (language or vision).
    Layers are selected by matching substrings in their parameter names.

    Notes
    -----
    - The Hadamard rotation is scaled by `1/sqrt(dim)` to preserve norm.
    - Shape compatibility is assumed based on common LoRA conventions:
      `lora_A` often has shape `[out_dim, rank]` and `lora_B` has `[rank, out_dim]`.
    """

    assert language_hidden_size or vision_hidden_size, (
        "At least one of language_hidden_size and vision_hidden_size must be provided"
    )

    rotated_adapter_weights = copy.deepcopy(peft_state_dict)

    rotation_language = None
    if language_hidden_size:
        rotation_language = get_hadamard_matrix(language_hidden_size) / torch.sqrt(
            torch.tensor(language_hidden_size)
        )

    rotation_vision = None
    if vision_hidden_size:
        rotation_vision = get_hadamard_matrix(vision_hidden_size) / torch.sqrt(
            torch.tensor(vision_hidden_size)
        )

    for layername, weight in peft_state_dict.items():
        if any(sub in layername for sub in layers_with_lhs_rotations) and "lora_A" in layername:
            if rotation_language is not None and language_layers_filter in layername:
                rotated_adapter_weights.update({layername: (rotation_language.T @ weight.T).T})
            elif rotation_vision is not None and vision_layers_filter in layername:
                rotated_adapter_weights.update({layername: (rotation_vision.T @ weight.T).T})
            else:
                raise Exception(f"Unable to apply left-hand-side rotation to lora_A in {layername} with weight shape {weight.shape}")
            logger.debug('SpinQuant: left-hand-side rotation applied to %s', layername)
        elif any(sub in layername for sub in layers_with_rhs_rotations) and "lora_B" in layername:
            if rotation_language is not None and language_layers_filter in layername:
                rotated_adapter_weights.update({layername: (weight.T @ rotation_language).T})
            elif rotation_vision is not None and vision_layers_filter in layername:
                rotated_adapter_weights.update({layername: (weight.T @ rotation_vision).T})
            else:
                raise Exception( f"Unable to apply right-hand-side rotation to lora_B in {layername} with weight shape {weight.shape}")
            logger.debug('SpinQuant: right-hand-side rotation applied to %s', layername)
        else:
            # No rotation applied for this layer; record this for telemetry.
            logger.info('SpinQuant: no rotation applied to %s', layername)


    # Ensure contiguous tensors
    rotated_adapter_weights = {k: (v.detach().contiguous()) for k, v in rotated_adapter_weights.items()}
    return rotated_adapter_weights


def _find_module_by_name_substring(model: torch.nn.Module, substring: str) -> Optional[Tuple[str, torch.nn.Module]]:
    """Return (name, module) whose qualified name contains the given substring (first match)."""
    for name, mod in model.named_modules():
        if substring in name:
            return name, mod
    return None

# -----------------------------
# (A) Capture per-layer norm parameters (weights & optional bias)
# -----------------------------

def capture_norm_fusion_factors(
    model: torch.nn.Module,
    layers_iterable: Iterable[torch.nn.Module],
    input_ln_attr: str = 'input_layernorm',
    post_ln_attr: str = 'post_attention_layernorm',
    final_norm_attr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Capture per-layer norm scales (and biases) prior to SpinQuant.

    Args:
      model: full model (used only to locate final norm via name substring)
      layers_iterable: iterable sequence of blocks (decoder/VLM blocks)
      input_ln_attr: attribute name of the *input* norm on each block (feeds q/k/v/qkv)
      post_ln_attr: attribute name of the *post-attn* norm on each block (feeds gate/up)
      final_norm_attr: if provided, locate the first module whose qualified name contains
                       this substring and record it as 'final_norm'

    Returns a dict:
      {
        'input_layernorm':        {layer_idx: {'weight': D_in,  'bias': beta_in or None}},
        'post_attention_layernorm':{layer_idx: {'weight': D_post,'bias': beta_post or None}},
        'final_norm':             {'weight': D_final, 'bias': beta_final or None}  # present only if found
      }
    """
    out: Dict[str, Any] = {'input_layernorm': {}, 'post_attention_layernorm': {}, 'final_norm': {}}

    for i, block in enumerate(layers_iterable):
        ln_in = getattr(block, input_ln_attr, None)
        if ln_in is not None and getattr(ln_in, 'weight', None) is not None:
            D = ln_in.weight.detach().clone()
            B = ln_in.bias.detach().clone() if hasattr(ln_in, 'bias') and ln_in.bias is not None else None
            out['input_layernorm'][i] = {'weight': D, 'bias': B}
        ln_post = getattr(block, post_ln_attr, None)
        if ln_post is not None and getattr(ln_post, 'weight', None) is not None:
            D = ln_post.weight.detach().clone()
            B = ln_post.bias.detach().clone() if hasattr(ln_post, 'bias') and ln_post.bias is not None else None
            out['post_attention_layernorm'][i] = {'weight': D, 'bias': B}

    if final_norm_attr:
        hit = _find_module_by_name_substring(model, final_norm_attr)
        if hit is not None:
            _, final_norm = hit
            if getattr(final_norm, 'weight', None) is not None:
                D = final_norm.weight.detach().clone()
                B = final_norm.bias.detach().clone() if hasattr(final_norm, 'bias') and final_norm.bias is not None else None
                out['final_norm']['weight'] = D
                out['final_norm']['bias'] = B

    return out

# -----------------------------
# (B) Apply norm column scaling to LoRA-A (single-stream API)
# -----------------------------

def apply_norm_scaling_to_lora(
    peft_state_dict: Mapping[str, torch.Tensor],
    factors: Dict[str, Any],
    layername_filter: str,
    norm_to_targets: Mapping[str, Sequence[str]] = None,
    layer_index_patterns: Sequence[str] = (r"layers\.(\d+)\.",),
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Fold norm scaling into LoRA-A using attach-point + substring filter (single stream).

    Selection mirrors your existing pattern:
        if any(sub in key for sub in all_lhs) and (layername_filter in key): scale A

    Args:
      peft_state_dict: adapter state dict
      factors: output of capture_norm_fusion_factors for this one stream
      layername_filter: substring that must appear in parameter key to belong to this stream
      norm_to_targets: dict mapping norm source -> tuple/list of attach substrings
                       e.g., {
                         'input_layernorm': ('q_proj','k_proj','v_proj','qkv'),
                         'post_attention_layernorm': ('gate_proj','up_proj'),
                       }
      layer_index_patterns: regexes to extract layer index from parameter key (group 1 = index)
      strict: raise on shape/index issues; if False, skip with warning

    Returns:
      New state dict with LoRA-A tensors column-scaled by the appropriate per-layer norm vector.
    """
    if norm_to_targets is None:
        norm_to_targets = {
            'input_layernorm': ('q_proj', 'k_proj', 'v_proj', 'qkv'),
            'post_attention_layernorm': ('gate_proj', 'up_proj'),
        }

    # Build the concatenated LHS set once
    all_lhs = tuple(sorted({a for targets in norm_to_targets.values() for a in targets}))
    updated = dict(peft_state_dict)

    def _which_norm_source(key: str) -> Optional[str]:
        for ns, targets in norm_to_targets.items():
            # require ".{target}." to avoid accidental partial matches
            if any(f".{t}." in key for t in targets):
                return ns
        return None

    for key, tensor in peft_state_dict.items():
        if not key.endswith('.lora_A.weight'):
            continue
        if not any(f".{sub}." in key for sub in all_lhs):
            continue
        if layername_filter and (layername_filter not in key):
            continue

        # Extract layer index
        layer_idx: Optional[int] = None
        for pat in layer_index_patterns:
            m = re.search(pat, key)
            if m:
                layer_idx = int(m.group(1))
                break
        if layer_idx is None:
            if strict:
                raise ValueError(f'Could not extract layer index from key: {key}')
            else:
                logger.warning('[norm-fold][warn] no layer index match for:', key)
                continue

        # Determine which norm source applies for this attach point
        norm_source = _which_norm_source(key)
        if norm_source is None:
            # Not mapped to any norm source per user mapping; skip
            continue

        entry = factors.get(norm_source, {}).get(layer_idx)
        if not entry:
            if strict:
                raise KeyError(f"Missing norm factors for {norm_source}[{layer_idx}] (key={key})")
            else:
                logger.warning(f"[norm-fold][warn] missing {norm_source}[{layer_idx}] for key={key}")
                continue

        D = entry['weight'].to(tensor.device).to(tensor.dtype)

        if tensor.dim() != 2:
            msg = f'Unexpected lora_A dim for {key}: {tuple(tensor.shape)}'
            if strict:
                raise ValueError(msg)
            else:
                logger.warning('[norm-fold][warn]', msg)
                continue

        _, in_features = tensor.shape
        if D.numel() != in_features:
            msg = f'Norm scale length mismatch for {key}: A {tuple(tensor.shape)} vs D {tuple(D.shape)}'
            if strict:
                raise ValueError(msg)
            else:
                logger.warning('[norm-fold][warn]', msg)
                continue

        # Column-wise scaling
        updated[key] = tensor * D

    return updated


# -----------------------------
# (C) AdaScale / quantsim helpers
# -----------------------------

class ONNXExportableModuleWithCache(torch.nn.Module):
    """
    Helper class to enable Torch JIT trace / ONNX export of HuggingFace models that produce and
    consume Cache objects.

    The wrapper always holds a *text decoder* plus its *LM-head*:
      - LLM : decoder = model.model (the CausalLM's backbone), lm_head = model.lm_head
      - VLM : decoder = model.model.language_model (text decoder only), lm_head = model.lm_head

    """

    def __init__(self, decoder, lm_head):
        super().__init__()
        self.model = decoder
        self.lm_head = lm_head

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped decoder when not found on this wrapper."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # pylint: disable=keyword-arg-before-vararg
    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        visual_pos_masks: torch.Tensor = None,
        deepstack_visual_embeds=None,
        *args,
        **kwargs
    ):
        """Redefine model forward to convert to/from Huggingface DynamicCache objects."""
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # `return_dict` (deprecated) / `num_logits_to_return` must not reach the decoder; drop them
        # before forwarding the remaining kwargs.
        kwargs.pop("return_dict", None)
        kwargs.pop("num_logits_to_return", None)

        lm_kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        # The HF text decoder requires exactly one of inputs_embeds / input_ids. AdaScale (LLM+VLM)
        # and the trace dummy feed `inputs_embeds`; only the text-only PPL eval feeds `input_ids`.
        if inputs_embeds is not None:
            lm_kwargs["inputs_embeds"] = inputs_embeds
        else:
            lm_kwargs["input_ids"] = input_ids
        # `visual_pos_masks` / `deepstack_visual_embeds` only exist on Qwen3-VL's text decoder;
        # pass them through only when actually provided (InternVL / LLM decoders lack these args).
        if deepstack_visual_embeds is not None:
            lm_kwargs["visual_pos_masks"] = visual_pos_masks
            lm_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds

        lm_kwargs = {**kwargs, **lm_kwargs}
        outputs = self.model(*args, **lm_kwargs)
        lm_logits = self.lm_head(outputs.last_hidden_state)
        new_past_key_values = outputs.past_key_values
        if isinstance(new_past_key_values, DynamicCache):
            new_past_key_values = new_past_key_values.to_legacy_cache()
        return lm_logits, new_past_key_values


def _set_blocks(full_model, decoder=None, lm_head=None, is_VLM=False):
    """Swap (QDQ) blocks back into the full HF model for PEFT eval / export. `decoder` is placed at
    the text-decoder slot (VLM: model.language_model; LLM: model); `lm_head`, when given, replaces
    the head. Either may be omitted to leave that block untouched. `is_VLM` selects the decoder
    slot."""
    if decoder is not None:
        if is_VLM:
            full_model.model.language_model = decoder
        else:
            full_model.model = decoder
    if lm_head is not None:
        full_model.lm_head = lm_head


# -----------------------------
# (D) Prompt-masking loss helpers (AdaScale on VLMs)
# -----------------------------

def get_assistant_header_token_ids(processor) -> list:
    """Token ids of the header that `add_generation_prompt=True` appends to the chat template
    (e.g. '<|im_start|>assistant\\n'), derived from the processor itself so it works for any
    model's template. Used to locate the prompt/response boundary for prompt-masking loss.

    Assumes `add_generation_prompt=True` only *appends* an assistant header to the templated
    string (it does not rewrite earlier turns)."""
    tokenizer = getattr(processor, "tokenizer", processor)
    probe = [{"role": "user", "content": "x"}]
    with_header = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    without_header = tokenizer.apply_chat_template(probe, add_generation_prompt=False, tokenize=False)
    assert with_header.startswith(without_header), (
        "Generation prompt is not a pure suffix "
    )
    assistant_header_text = with_header[len(without_header):]  # the appended suffix only
    return tokenizer(assistant_header_text, add_special_tokens=False)["input_ids"]


def find_response_start_index(input_ids: torch.Tensor, assistant_header_token_ids: Sequence[int]) -> list:
    """Index in `input_ids` where the assistant response begins. Our calibration datasets are
    single-turn, and the dataloaders template the (system, user, assistant) messages with
    `add_generation_prompt=True`. That yields exactly two assistant headers: the first wraps the
    real response, the second is the trailing generation prompt. The response begins right after
    the first header."""
    header = list(assistant_header_token_ids)
    header_len = len(header)
    response_starts = []
    for ids in input_ids.tolist():
        header_positions = [
            i for i in range(len(ids) - header_len + 1)
            if ids[i:i + header_len] == header
        ]
        first_pos, second_pos = header_positions
        assert first_pos < second_pos, (
            f"Assistant header positions are not strictly increasing: {header_positions}"
        )
        assert first_pos + header_len <= second_pos, (
            "Malformed sample: first assistant header overlaps or extends past the second header"
        )
        tail_after_second_header = ids[second_pos + header_len:]
        assert len(tail_after_second_header) == 0, (
            "Expected second assistant header to be the trailing generation prompt at the end "
            "of the sequence, but found extra tokens after it"
        )
        response_starts.append(first_pos + header_len)

    return response_starts
