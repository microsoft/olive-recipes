# SAM `modeling_sam.py` API Diff: transformers 4.51.3 vs 4.56.2

> **Environment mapping:**
> - `sam` conda env → transformers **4.51.3** (older)
> - `sam-new` conda env → transformers **4.56.2** (newer)

---

## 1. Import Changes

| Old (4.51.3) | New (4.56.2) |
|---|---|
| `from typing import Optional, Tuple, Union` | `from typing import Callable, Optional, Union` (dropped `Tuple`, uses built-in `tuple`) |
| `import torch.utils.checkpoint` | Removed (replaced by `GradientCheckpointingLayer`) |
| `from ...modeling_utils import PreTrainedModel` | `from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel` |
| `add_start_docstrings, can_return_tuple, replace_return_docstrings` | `auto_docstring` (unified doc decorator) |
| — | Added `from ...processing_utils import Unpack` |
| — | Added `from ...modeling_layers import GradientCheckpointingLayer` |
| — | Added `from transformers.utils.generic import OutputRecorder, TransformersKwargs, check_model_inputs` |

---

## 2. Type Annotation Modernization

All `Tuple[...]` replaced with Python built-in `tuple[...]` throughout the file:

```python
# Old
Optional[Tuple[torch.FloatTensor, ...]]
# New
Optional[tuple[torch.FloatTensor, ...]]
```

---

## 3. `SamLayerNorm` Refactored

```python
# Old: inherits nn.Module, manages weight/bias manually
class SamLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

# New: inherits nn.LayerNorm, delegates to super().forward()
class SamLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, *, eps=1e-6, data_format="channels_last", **kwargs):
        super().__init__(normalized_shape, eps=eps, **kwargs)
```

The `channels_first` forward implementation also changed — old version manually computes mean/std, new version uses `permute` + `super().forward()`.

---

## 4. Attention Mechanism Overhaul (Most Important Change)

### 4a. Unified Attention Interface

- **Old**: Two separate classes `SamAttention` and `SamSdpaAttention`, dispatched via a dict:
  ```python
  SAM_ATTENTION_CLASSES = {"eager": SamAttention, "sdpa": SamSdpaAttention}
  ```
- **New**: `SamSdpaAttention` is **deleted**. Only `SamAttention` remains, dispatching via `ALL_ATTENTION_FUNCTIONS`:
  ```python
  attention_interface: Callable = eager_attention_forward
  if self.config._attn_implementation != "eager":
      attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
  ```

### 4b. New Global Function `eager_attention_forward`

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    ...
    return attn_output, attn_weights
```

### 4c. `SamAttention.forward` Signature Change

```python
# Old — returns a single Tensor
def forward(self, query, key, value, attention_similarity=None) -> Tensor:
    ...
    return out

# New — returns tuple (output, attn_weights), accepts **kwargs
def forward(self, query, key, value, attention_similarity=None,
            **kwargs: Unpack[TransformersKwargs]):
    ...
    return attn_output, attn_weights
```

### 4d. `_recombine_heads` Internal Change

```python
# Old: transpose then reshape
batch, n_heads, n_tokens, c_per_head = hidden_states.shape
hidden_states = hidden_states.transpose(1, 2)

# New: already in correct dim order, reshape directly
batch, n_tokens, n_heads, c_per_head = hidden_states.shape
```

### 4e. New Attributes on `SamAttention`

```python
self.config = config
self.scaling = (self.internal_dim // config.num_attention_heads) ** -0.5
self.is_causal = False
```

---

## 5. `SamVisionAttention.forward` Return Value Change

```python
# Old — conditional return
def forward(self, hidden_states, output_attentions=False) -> torch.Tensor:
    if output_attentions:
        outputs = (attn_output, attn_weights)
    else:
        outputs = (attn_output, None)
    return outputs

# New — always returns both values
def forward(self, hidden_states, output_attentions=None) -> tuple[torch.Tensor, torch.Tensor]:
    return attn_output, attn_weights
```

---

## 6. `SamVisionLayer` Changes

```python
# Old
class SamVisionLayer(nn.Module):
    def forward(self, hidden_states, output_attentions=False) -> Tuple[torch.FloatTensor]:
        ...
        return outputs  # tuple of (hidden_states,) or (hidden_states, attn_weights)

# New
class SamVisionLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states) -> tuple[torch.FloatTensor]:
        ...
        return hidden_states  # returns tensor directly, no longer a tuple
```

The `output_attentions` parameter is **removed** — the layer is no longer responsible for collecting attention outputs.

---

## 7. `SamVisionEncoder` Changes

```python
# Old: inherits nn.Module
class SamVisionEncoder(nn.Module):
    def forward(self, pixel_values, output_attentions=None, output_hidden_states=None):
        # Manually manages all_hidden_states, all_self_attentions
        # Manually handles gradient_checkpointing

# New: inherits SamPreTrainedModel
class SamVisionEncoder(SamPreTrainedModel):
    _can_record_outputs = {"hidden_states": SamVisionLayer, "attentions": SamVisionAttention}
    def forward(self, pixel_values, **kwargs: Unpack[TransformersKwargs]):
        # Simplified forward; hidden_states/attentions collected automatically
        # by the framework via OutputRecorder
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
```

**Key difference**: Collection logic for `output_attentions` and `output_hidden_states` is moved from model code to the framework level (via `_can_record_outputs` + `OutputRecorder`).

---

## 8. `SamPreTrainedModel` Changes

```python
# Old
class SamPreTrainedModel(PreTrainedModel):
    config_class = SamConfig
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            ...

# New — simplified, most init logic delegated to parent
class SamPreTrainedModel(PreTrainedModel):
    config: SamConfig  # type annotation instead of config_class
    def _init_weights(self, module):
        super()._init_weights(module)  # delegate to parent
        # Only handles SAM-specific: rel_pos and pos_embed
```

---

## 9. `SamTwoWayTransformer.forward` Signature and Return Value

```python
# Old — accepts output_attentions/output_hidden_states, returns 3 values
def forward(self, ..., output_attentions=None,
            output_hidden_states=None) -> Union[Tuple, BaseModelOutput]:
    ...
    return queries, keys, all_attentions

# New — accepts **kwargs only, returns 2 values
def forward(self, ...,
            **kwargs: Unpack[TransformersKwargs]) -> Union[tuple, BaseModelOutput]:
    ...
    return queries, keys
```

---

## 10. `SamTwoWayAttentionBlock.forward` Signature and Return Value

```python
# Old
def forward(self, ..., output_attentions: bool = False):
    ...
    outputs = (queries, keys)
    if output_attentions:
        outputs = outputs + (attn_out,)
    else:
        outputs = outputs + (None,)
    return outputs

# New
def forward(self, ..., **kwargs: Unpack[TransformersKwargs]):
    ...
    return queries, keys, attn_out  # always returns all three
```

---

## 11. `SamMaskDecoder.forward` Signature and Return Value

```python
# Old — returns 3-tuple
def forward(self, ..., output_attentions=None) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
    return (masks, iou_pred, attentions_or_None)

# New — returns 2-tuple
def forward(self, ...) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    return masks, iou_pred
```

The `output_attentions` parameter is **removed**. Mask decoder attention is now collected via:
```python
_can_record_outputs = {"mask_decoder_attentions": OutputRecorder(SamTwoWayAttentionBlock, index=2)}
```

---

## 12. `SamPromptEncoder.__init__` Signature Change

```python
# Old
def __init__(self, config: SamPromptEncoderConfig):

# New
def __init__(self, config: SamConfig):  # accepts full SamConfig
```

The default handling for `sparse_embeddings` also changed:

```python
# Old — creates a zero tensor as default
if sparse_embeddings is None:
    sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

# New — no default value created; None is handled downstream in mask_decoder
```

Additional change in `sparse_prompt_embeddings` check:
```python
# Old
if sparse_prompt_embeddings.sum().item() != 0:
# New
if sparse_prompt_embeddings is not None:
```

---

## 13. `SamModel.forward` Signature

```python
# Old
def forward(self, ..., output_attentions=None, output_hidden_states=None, **kwargs):

# New
def forward(self, ..., **kwargs: Unpack[TransformersKwargs]):
```

`output_attentions` and `output_hidden_states` are **no longer explicit parameters** — they are passed through `TransformersKwargs`.

---

## 14. `SamVisionModel` Changes

```python
# Old
@add_start_docstrings(...)
class SamVisionModel(SamPreTrainedModel):
    config_class = SamVisionConfig
    def forward(self, pixel_values, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        return self.vision_encoder(pixel_values, output_attentions=...,
                                   output_hidden_states=..., return_dict=...)

# New
@auto_docstring(...)
class SamVisionModel(SamPreTrainedModel):
    config: SamVisionConfig
    def forward(self, pixel_values, **kwargs: Unpack[TransformersKwargs]):
        return self.vision_encoder(pixel_values, **kwargs)
```

---

## 15. Docstring Infrastructure

| Old (4.51.3) | New (4.56.2) |
|---|---|
| `SAM_START_DOCSTRING` (module-level string constant) | Removed |
| `SAM_INPUTS_DOCSTRING` (module-level string constant) | Removed |
| `SAM_VISION_INPUTS_DOCSTRING` (module-level string constant) | Removed |
| `@add_start_docstrings(...)` | `@auto_docstring(custom_intro=...)` |
| `@add_start_docstrings_to_model_forward(...)` | `@auto_docstring` |
| `@replace_return_docstrings(...)` | Removed |
| `@can_return_tuple` | `@check_model_inputs` |
| `_CONFIG_FOR_DOC`, `_CHECKPOINT_FOR_DOC` constants | Removed |

---

## Summary Table

| Dimension | Impact |
|---|---|
| **Return values** | Many functions changed from 3-tuple to 2-tuple (attentions no longer returned inline). `SamVisionLayer.forward` changed from returning a tuple to returning a tensor directly. |
| **Function parameters** | `output_attentions` / `output_hidden_states` changed from explicit parameters to being passed via `**kwargs: Unpack[TransformersKwargs]`. |
| **Attention classes** | `SamSdpaAttention` deleted, unified into a single `SamAttention` class with pluggable attention backends. |
| **Inheritance hierarchy** | `SamVisionEncoder`: `nn.Module` → `SamPreTrainedModel`; `SamVisionLayer`: `nn.Module` → `GradientCheckpointingLayer`; `SamLayerNorm`: `nn.Module` → `nn.LayerNorm`. |
| **Output collection** | Moved from manual collection in forward methods to framework-level `OutputRecorder` / `_can_record_outputs`. |
| **ONNX / Export** | `sparse_embeddings` None-handling logic changed — may affect model export pipelines. |
| **Weight init** | Simplified; most logic delegated to `PreTrainedModel._init_weights()` parent. |
