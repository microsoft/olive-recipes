"""
Generic ONNX Model WebGPU Fix for Combined QKV Projection Issues

PROBLEM SUMMARY:
================
WebGPU-converted ONNX models with combined qkv_proj structures exhibit a critical
architecture mismatch:

1. GroupQueryAttention nodes use K, V projections from PREVIOUS layers instead of
   the same layer
2. GroupQueryAttention Q input receives the full combined qkv_proj output instead
   of just the Q portion
3. This causes dimension mismatch: o_proj expects specific K dimension but receives
   mismatched output from GroupQueryAttention

EXAMPLES:
- DeepSeek-R1-Distill-Qwen-1.5B: qkv_proj=2048, Q=1536, K=256, V=256
- Llama-3.2-1B: qkv_proj=3072, Q=2048, K=512, V=512

SOLUTION OVERVIEW:
==================
For each affected layer, we:
1. Extract Q from qkv_proj[0:Q_dim]
2. Extract K from qkv_proj[Q_dim:Q_dim+K_dim]
3. Extract V from qkv_proj[Q_dim+K_dim:total_dim]
4. Update GroupQueryAttention to use extracted tensors
5. Ensure all new tensors match model precision
6. Use proper ONNX Slice syntax (axes as input, not attribute)
"""

import json
import sys

import onnx
from onnx import helper


def auto_detect_layers_and_dims(model_path):
    """
    Auto-detect which layers have combined qkv_proj and their dimensions.

    Returns: (layers_to_fix, q_dim, k_dim, v_dim) or (None, None, None, None) if not found
    """
    try:
        model = onnx.load(model_path, load_external_data=False)
        graph = model.graph

        layers_to_fix = []
        qkv_dim = None

        # Find layers with qkv_proj
        for i in range(64):
            has_qkv = False
            for node in graph.node:
                if f"layers.{i}/attn" in node.name and "qkv_proj" in node.name:
                    has_qkv = True
                    if qkv_dim is None:
                        # Get qkv_proj output dimension
                        for vi in graph.value_info:
                            if f"layers.{i}/attn/qkv_proj" in vi.name and "output" in vi.name:
                                dims = vi.type.tensor_type.shape.dim
                                qkv_dim = dims[-1].dim_value

            if has_qkv:
                layers_to_fix.append(i)

        if not layers_to_fix or qkv_dim is None:
            return None, None, None, None

        # Get o_proj K dimension to infer Q_dim
        o_proj_k = None
        for i in layers_to_fix:
            for node in graph.node:
                if node.name == f"/model/layers.{i}/attn/o_proj/MatMulNBits":
                    for attr in node.attribute:
                        if attr.type == 2 and attr.name == "K":
                            o_proj_k = attr.i
                    break
            if o_proj_k:
                break

        if qkv_dim and o_proj_k:
            q_dim = o_proj_k
            remaining = qkv_dim - q_dim
            k_dim = remaining // 2
            v_dim = remaining - k_dim
            return layers_to_fix, q_dim, k_dim, v_dim

        return None, None, None, None
    except Exception:
        return None, None, None, None


def fix_webgpu_qkv_model(model_path, layers_to_fix=None, q_dim=None, k_dim=None, v_dim=None, auto_detect=True):
    """
    Generic fix for WebGPU ONNX models with combined qkv_proj dimension mismatch.

    Parameters:
    -----------
    model_path : str
        Path to the ONNX model file
    layers_to_fix : list
        Layer IDs to fix (auto-detected if None)
    q_dim : int
        Query dimension (auto-detected if None)
    k_dim : int
        Key dimension (auto-detected if None)
    v_dim : int
        Value dimension (auto-detected if None)
    auto_detect : bool
        If True, auto-detect layers and dimensions (overrides manual params)

    Returns:
    --------
    bool : True if successful, False otherwise
    """

    print("=" * 70)
    print("Generic WebGPU ONNX QKV Model Fixer")
    print("=" * 70)

    try:
        # Load model
        print(f"\n[1/4] Loading model from {model_path}...")
        model = onnx.load(model_path, load_external_data=False)
        graph = model.graph
        print(f"  ✓ Model loaded successfully")
        print(f"  - IR Version: {model.ir_version}")
        print(f"  - Opset: {model.opset_import[0].version if model.opset_import else 'unknown'}")

        # Auto-detect if enabled
        if auto_detect:
            print(f"\n[2/4] Auto-detecting layers and dimensions...")
            det_layers, det_q, det_k, det_v = auto_detect_layers_and_dims(model_path)
            if det_layers:
                layers_to_fix = det_layers
                q_dim = det_q
                k_dim = det_k
                v_dim = det_v
                print(f"  ✓ Detected layers: {layers_to_fix}")
                print(f"  ✓ Detected dimensions: Q={q_dim}, K={k_dim}, V={v_dim}")

        if not layers_to_fix or not q_dim or not k_dim or not v_dim:
            print(f"  ✗ Failed to detect or specify layers and dimensions")
            return False

        total_dim = q_dim + k_dim + v_dim
        print(f"\n[3/4] Setting up Slice operations...")
        print(f"  • Total QKV dim: {total_dim} = {q_dim} + {k_dim} + {v_dim}")

        # Create required constants for Slice operations
        constants = {
            "const_0": 0,
            f"const_{q_dim}": q_dim,
            f"const_{q_dim + k_dim}": q_dim + k_dim,
            f"const_{total_dim}": total_dim,
            "const_axes_2": [2],
        }

        # Add constants to graph
        for const_name, const_value in constants.items():
            if not any(init.name == const_name for init in graph.initializer):
                if const_name == "const_axes_2":
                    tensor = helper.make_tensor(const_name, onnx.TensorProto.INT64, [1], const_value)
                else:
                    tensor = helper.make_tensor(const_name, onnx.TensorProto.INT64, [1], [const_value])
                graph.initializer.append(tensor)

        # Fix each layer
        slices_added = 0
        for layer_id in layers_to_fix:
            # Auto-detect qkv_proj output node (could be Add or MatMulNBits)
            qkv_output = None
            for node in graph.node:
                if node.name == f"/model/layers.{layer_id}/attn/qkv_proj/Add":
                    qkv_output = f"/model/layers.{layer_id}/attn/qkv_proj/Add/output_0"
                    break

            if not qkv_output:
                # Fall back to MatMulNBits if no Add node
                for node in graph.node:
                    if node.name == f"/model/layers.{layer_id}/attn/qkv_proj/MatMulNBits":
                        qkv_output = f"/model/layers.{layer_id}/attn/qkv_proj/MatMulNBits/output_0"
                        break

            if not qkv_output:
                print(f"  ✗ Could not find qkv_proj output for layer {layer_id}")
                return False

            # Find data type from qkv_proj output
            dtype = onnx.TensorProto.FLOAT16
            for vi in graph.value_info:
                if f"layers.{layer_id}/attn/qkv_proj" in vi.name and "output" in vi.name:
                    dtype = vi.type.tensor_type.elem_type
                    break

            # Q extraction: [0:q_dim]
            slice_q = helper.make_node(
                "Slice",
                inputs=[qkv_output, "const_0", f"const_{q_dim}", "const_axes_2"],
                outputs=[f"/model/layers.{layer_id}/attn/q_proj_extracted/output_0"],
                name=f"/model/layers.{layer_id}/attn/q_proj_extracted/Slice",
            )

            # K extraction: [q_dim:q_dim+k_dim]
            slice_k = helper.make_node(
                "Slice",
                inputs=[qkv_output, f"const_{q_dim}", f"const_{q_dim + k_dim}", "const_axes_2"],
                outputs=[f"/model/layers.{layer_id}/attn/k_proj_extracted/output_0"],
                name=f"/model/layers.{layer_id}/attn/k_proj_extracted/Slice",
            )

            # V extraction: [q_dim+k_dim:total]
            slice_v = helper.make_node(
                "Slice",
                inputs=[qkv_output, f"const_{q_dim + k_dim}", f"const_{total_dim}", "const_axes_2"],
                outputs=[f"/model/layers.{layer_id}/attn/v_proj_extracted/output_0"],
                name=f"/model/layers.{layer_id}/attn/v_proj_extracted/Slice",
            )

            graph.node.extend([slice_q, slice_k, slice_v])
            slices_added += 3

            # Add value_info for extracted tensors
            q_info = helper.make_tensor_value_info(
                f"/model/layers.{layer_id}/attn/q_proj_extracted/output_0",
                dtype,
                ["batch_size", "sequence_length", q_dim],
            )
            k_info = helper.make_tensor_value_info(
                f"/model/layers.{layer_id}/attn/k_proj_extracted/output_0",
                dtype,
                ["batch_size", "sequence_length", k_dim],
            )
            v_info = helper.make_tensor_value_info(
                f"/model/layers.{layer_id}/attn/v_proj_extracted/output_0",
                dtype,
                ["batch_size", "sequence_length", v_dim],
            )
            graph.value_info.extend([q_info, k_info, v_info])

            # Update GroupQueryAttention inputs
            for node in graph.node:
                if node.name == f"/model/layers.{layer_id}/attn/GroupQueryAttention":
                    node.input[0] = f"/model/layers.{layer_id}/attn/q_proj_extracted/output_0"
                    node.input[1] = f"/model/layers.{layer_id}/attn/k_proj_extracted/output_0"
                    node.input[2] = f"/model/layers.{layer_id}/attn/v_proj_extracted/output_0"
                    break

        print(f"  ✓ Added {slices_added} Slice nodes across {len(layers_to_fix)} layers")
        print(f"  ✓ Updated {len(layers_to_fix)} GroupQueryAttention nodes")

        # Save fixed model
        print(f"\n[4/4] Saving fixed model...")
        onnx.save(model, model_path)
        print(f"  ✓ Model saved successfully")

        print("\n" + "=" * 70)
        print("FIX COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nSummary of Changes:")
        print(f"  • Fixed {len(layers_to_fix)} layers: {layers_to_fix}")
        print(f"  • QKV dimensions: Q={q_dim}, K={k_dim}, V={v_dim}")
        print(f"  • Added {slices_added} Slice nodes for Q/K/V extraction")
        print(f"  • Corrected GroupQueryAttention layer cross-references")
        print(f"  • Ensured precision consistency for all new tensors")
        print(f"  • Updated Slice syntax for ONNX opset 21 compatibility")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def verify_fix(model_path, verbose=False, layers_to_fix=None):
    """
    Verify that the fix was applied correctly.

    Parameters:
    -----------
    model_path : str
        Path to the fixed ONNX model
    verbose : bool
        Print detailed information
    layers_to_fix : list
        Specific layers to verify (auto-detected if None)

    Returns:
    --------
    bool : True if fix is verified, False otherwise
    """

    print("\nVerifying model fix...")

    try:
        model = onnx.load(model_path, load_external_data=False)
        graph = model.graph

        # Auto-detect layers if not provided
        if layers_to_fix is None:
            det_result = auto_detect_layers_and_dims(model_path)
            if det_result and det_result[0]:
                layers_to_fix = det_result[0]
            else:
                print("  ✗ No layers detected - model may not need fixing or has unknown structure")
                return False

        if not layers_to_fix or not isinstance(layers_to_fix, list):
            print("  ✗ Invalid layers list")
            return False

        all_correct = True

        for layer_id in layers_to_fix:
            # Check Slice nodes exist
            slice_nodes = [
                n
                for n in graph.node
                if f"layers.{layer_id}" in n.name and "Slice" in n.name and "proj_extracted" in n.name
            ]

            if len(slice_nodes) != 3:
                print(f"  ✗ Layer {layer_id}: Expected 3 Slice nodes, found {len(slice_nodes)}")
                all_correct = False
                continue

            # Check GroupQueryAttention inputs
            gqa_node = next(
                (n for n in graph.node if n.name == f"/model/layers.{layer_id}/attn/GroupQueryAttention"), None
            )

            if not gqa_node:
                print(f"  ✗ Layer {layer_id}: GroupQueryAttention node not found")
                all_correct = False
                continue

            # Verify inputs point to extracted tensors
            q_correct = gqa_node.input[0] == f"/model/layers.{layer_id}/attn/q_proj_extracted/output_0"
            k_correct = gqa_node.input[1] == f"/model/layers.{layer_id}/attn/k_proj_extracted/output_0"
            v_correct = gqa_node.input[2] == f"/model/layers.{layer_id}/attn/v_proj_extracted/output_0"

            if q_correct and k_correct and v_correct:
                if verbose:
                    print(f"  ✓ Layer {layer_id}: All checks passed")
            else:
                print(f"  ✗ Layer {layer_id}: GroupQueryAttention inputs incorrect")
                all_correct = False

        if all_correct:
            print("  ✓ All verifications passed!")

        return all_correct

    except Exception as e:
        print(f"  ✗ Verification failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Usage:
    # python fix_onnx_model.py [model_path]                    (auto-detect all)
    # python fix_onnx_model.py [model_path] --verify            (verify existing fix)
    # python fix_onnx_model.py [model_path] --config config.json (use config file)

    model_path = "./model/model.onnx"
    verify_only = False
    config_file = None

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if "--verify" in sys.argv:
        verify_only = True

    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_file = sys.argv[idx + 1]

    if verify_only:
        verify_fix(model_path, verbose=True)
        sys.exit(0)

    # Load config if provided
    q_dim = k_dim = v_dim = layers = None
    if config_file:
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                layers = config.get("layers_to_fix")
                q_dim = config.get("q_dim")
                k_dim = config.get("k_dim")
                v_dim = config.get("v_dim")
                print(f"Loaded config from {config_file}")
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")

    success = fix_webgpu_qkv_model(
        model_path,
        layers_to_fix=layers,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        auto_detect=True,  # Always auto-detect if values not provided
    )

    if success:
        verify_fix(model_path, verbose=True)
        sys.exit(0)
    else:
        sys.exit(1)
