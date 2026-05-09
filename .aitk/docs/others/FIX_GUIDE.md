# Generic WebGPU ONNX Model QKV Fix Guide

## Problem Description

WebGPU-converted ONNX models (DeepSeek, Llama, and others) with combined qkv_proj structures develop a critical dimension mismatch error in specific layers:

```
Node (/model/layers.X/attn/o_proj/MatMulNBits) Op (MatMulNBits) 
[ShapeInferenceError] Incompatible dimensions for matrix multiplication
```

### Root Cause

These layers have a **combined qkv_proj** structure (Q, K, V packed into one output), but the GroupQueryAttention operation was misconfigured:

| Issue | Problem |
|-------|---------|
| **Q input** | Receiving full 2048-dim qkv output instead of just Q (1536 dims) |
| **K input** | Using K from previous layer instead of current layer (256 dims from wrong source) |
| **V input** | Using V from previous layer instead of current layer (256 dims from wrong source) |
| **Result** | GroupQueryAttention produces mismatched output → o_proj fails |

### Layer Structure

Different models have this issue in different layers:

| Model | Layers with combined qkv_proj | Total QKV | Q | K | V |
|-------|-------------------------------|-----------|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 0, 6, 8, 12, 25, 26, 27 | 2048 | 1536 | 256 | 256 |
| Llama-3.2-1B | 2, 5, 6, 8, 10, 13 | 3072 | 2048 | 512 | 512 |

The `fix_onnx_model.py` script auto-detects this information automatically.

## Solution

For each affected layer, extract Q, K, V from the combined qkv_proj using Slice operations:

```
qkv_proj output (total_qkv dims):
  [0:q_dim]                → Q dimensions
  [q_dim:q_dim+k_dim]      → K dimensions  
  [q_dim+k_dim:total_qkv]  → V dimensions

GroupQueryAttention uses extracted Q, K, V → output matches o_proj expectations
```

**Example dimensions:**
- **DeepSeek:** [0:1536] Q, [1536:1792] K, [1792:2048] V
- **Llama:** [0:2048] Q, [2048:2560] K, [2560:3072] V

## Implementation

### Quick Start (Auto-Detect)

The script automatically detects affected layers and dimensions:

```bash
# From the model directory
cd ./model

# Run the fix (auto-detects everything)
python ../fix_onnx_model.py model.onnx

# Verify the fix
python ../fix_onnx_model.py model.onnx --verify
```

### Using Configuration File

For reproducibility or multiple models, create a `config.json`:

```json
{
  "layers_to_fix": [0, 6, 8, 12, 25, 26, 27],
  "q_dim": 1536,
  "k_dim": 256,
  "v_dim": 256
}
```

Then run:
```bash
python fix_onnx_model.py model.onnx --config config.json
```

### Examples for Common Models

**DeepSeek-R1-Distill-Qwen-1.5B config.json:**
```json
{
  "layers_to_fix": [0, 6, 8, 12, 25, 26, 27],
  "q_dim": 1536,
  "k_dim": 256,
  "v_dim": 256
}
```

**Llama-3.2-1B config.json:**
```json
{
  "layers_to_fix": [2, 5, 6, 8, 10, 13],
  "q_dim": 2048,
  "k_dim": 512,
  "v_dim": 512
}
```

### Manual Implementation (Advanced)

If you need to integrate this into your own code:

```python
from fix_onnx_model import fix_webgpu_qkv_model, verify_fix

# Auto-detect (recommended)
fix_webgpu_qkv_model('model.onnx')

# Or with explicit parameters
fix_webgpu_qkv_model(
    'model.onnx',
    layers_to_fix=[2, 5, 6, 8, 10, 13],  # Llama layers
    q_dim=2048,
    k_dim=512,
    v_dim=512,
    auto_detect=False  # Use provided values only
)

# Verify
verify_fix('model.onnx', verbose=True)
```

## Key Technical Details

### ONNX Slice Syntax

The `Slice` operator (opset 21) takes inputs in this order:
```
Slice(data, starts, ends, [axes], [steps])
```

- **data:** Input tensor to slice
- **starts:** Tensor with starting indices
- **ends:** Tensor with ending indices  
- **axes:** Tensor specifying which axes to slice (e.g., [2] for axis 2)
- **steps:** (optional) Step size for each axis

**Important:** Pass `axes` as an input tensor, NOT as an attribute (common mistake with older ONNX versions).

### Data Type Consistency

All new tensors must be **FLOAT16** to match:
- Input: `qkv_proj/Add/output_0` (FLOAT16)
- Output: `GroupQueryAttention/output_0` (FLOAT16)
- Subsequent layers expect FLOAT16 inputs

### Dimension Breakdown

The exact dimensions depend on your model's architecture:

**DeepSeek-R1-Distill-Qwen-1.5B:**
- num_heads=12, kv_num_heads=2, head_dim=128
- Q: 12 × 128 = 1536
- K: 2 × 128 = 256  
- V: 2 × 128 = 256
- Total: 1536 + 256 + 256 = 2048

**Llama-3.2-1B:**
- num_heads=32, kv_num_heads=8, head_dim=64
- Q: 32 × 64 = 2048
- K: 8 × 64 = 512
- V: 8 × 64 = 512
- Total: 2048 + 512 + 512 = 3072

To find these for any model:
```python
import onnx

model = onnx.load('model.onnx', load_external_data=False)
for vi in model.graph.value_info:
    if 'layers.0/attn/qkv_proj' in vi.name and 'output' in vi.name:
        qkv_dim = vi.type.tensor_type.shape.dim[-1].dim_value
        print(f"Total QKV dimension: {qkv_dim}")
        break

for node in model.graph.node:
    if 'layers.0/attn/o_proj' in node.name:
        for attr in node.attribute:
            if attr.name == 'K':
                print(f"Q dimension (from o_proj K): {attr.i}")
        break
```

## Verification

After applying the fix, verify that:

```python
import onnx

model = onnx.load('model.onnx', load_external_data=False)
layers_to_check = [0, 6, 8, 12, 25, 26, 27]  # Or your model's layers

for layer_id in layers_to_check:
    for node in model.graph.node:
        if node.name == f'/model/layers.{layer_id}/attn/GroupQueryAttention':
            print(f"Layer {layer_id}:")
            print(f"  Q: {node.input[0]}")     # Should be q_proj_extracted
            print(f"  K: {node.input[1]}")     # Should be k_proj_extracted
            print(f"  V: {node.input[2]}")     # Should be v_proj_extracted
            break
```

Expected pattern for fixed model:
```
Layer 0:
  Q: /model/layers.0/attn/q_proj_extracted/output_0
  K: /model/layers.0/attn/k_proj_extracted/output_0
  V: /model/layers.0/attn/v_proj_extracted/output_0
```

The script's `--verify` flag does this automatically:
```bash
python fix_onnx_model.py model.onnx --verify
```

## Usage Example

### DeepSeek-R1-Distill-Qwen-1.5B

```bash
cd C:\path\to\deepseek\model
python fix_onnx_model.py model/model.onnx
```

### Llama-3.2-1B  

```bash
cd C:\path\to\llama\model
python fix_onnx_model.py model/model.onnx
```

Both commands auto-detect layers and dimensions automatically. After the fix, your inference notebooks should work without shape inference errors:

```python
import onnxruntime_genai as og

# Model now loads successfully
model = og.Model('./model')
tokenizer = og.Tokenizer(model)

# Inference works correctly
generator = og.Generator(model, params)
```

## Detecting This Issue

If your WebGPU-converted model fails with shape inference errors, you can check if it has this issue:

```python
import onnx

model = onnx.load('model.onnx', load_external_data=False)

print("=== Checking for QKV cross-layer references ===")
affected_layers = []

for i in range(64):
    gqa_node = None
    for node in model.graph.node:
        if node.name == f'/model/layers.{i}/attn/GroupQueryAttention':
            gqa_node = node
            break
    
    if not gqa_node:
        continue
    
    has_qkv = any(f'layers.{i}/attn' in n.name and 'qkv_proj' in n.name 
                  for n in model.graph.node)
    
    if has_qkv:
        # Check if K/V come from different layers
        k_input = gqa_node.input[1]
        v_input = gqa_node.input[2]
        
        if f'layers.{i}' not in k_input or f'layers.{i}' not in v_input:
            print(f"  ✗ Layer {i}: Cross-layer reference detected")
            affected_layers.append(i)

if affected_layers:
    print(f"\nFix required for layers: {affected_layers}")
else:
    print("\nNo cross-layer references detected - model may not need fixing")
```

Typical output for affected models:
```
✗ Layer 2: Cross-layer reference detected
✗ Layer 5: Cross-layer reference detected
✗ Layer 6: Cross-layer reference detected
...
Fix required for layers: [2, 5, 6, 8, 10, 13]
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Unrecognized attribute: axes for operator Slice` | Ensure `axes` is passed as an input tensor, not an attribute (automatic in script) |
| `Type (tensor(float)) does not match expected type (tensor(float16))` | Verify all new tensors use correct data type - script auto-detects this |
| `Incompatible dimensions for matrix multiplication` | Confirm Slice indices match your model's dimensions (script auto-detects) |
| Model still fails after fix | Run with `--verify` flag to check all layers were processed correctly |
| Auto-detection doesn't work | Provide explicit config with `--config` flag |

## Supported Models

This fix has been tested on:
- ✅ DeepSeek-R1-Distill-Qwen-1.5B
- ✅ Llama-3.2-1B-Instruct
- ✅ Other WebGPU-converted models with similar cross-layer QKV issues

If you test this on other models, please note that auto-detection handles most cases. For models with non-standard structures, use the config file approach.

## References

- ONNX Slice operator: https://onnx.ai/onnx/operators/onnx__Slice.html
- ONNX spec: https://onnx.ai/onnx/
- DeepSeek-R1 Model: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Llama-3.2 Model: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- WebGPU ONNX Runtime: https://onnxruntime.ai/docs/execution-providers/web-gpu-execution-provider.html
- ONNX Runtime GenAI: https://github.com/microsoft/onnxruntime-genai
