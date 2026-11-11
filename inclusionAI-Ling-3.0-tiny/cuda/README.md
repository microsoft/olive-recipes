# Ling-3.0-tiny (CUDA)

These recipes export [`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
and [`inclusionAI/Ling-3.0-tiny-fp8`](https://huggingface.co/inclusionAI/Ling-3.0-tiny-fp8)
as ONNX Runtime GenAI packages for the CUDA execution provider using:

- Kimi Delta Attention (KDA) layers via `com.microsoft.VarlenCausalConvWithState`, `com.microsoft.GatedDeltaNet` (`gate_activation="sigmoid"`, `gate_lower_bound=-5.0`, per-key-dim decay), and `com.microsoft.GatedRMSNorm`;
- Multi-Head Latent Attention (MLA) layers via `com.microsoft.PagedAttention` (`head_size=192`, 256-token page blocks) with optional calibrated INT8 per-channel KV cache; and
- Sparse MoE layers (128 routed experts, top-8 `noaux_tc` sigmoid group-limited routing + 1 shared expert) via `com.microsoft.MoE` / `com.microsoft.QMoE`.

## Recipes

1. **BF16 baseline (`inclusionAI-Ling-3.0-tiny_cuda_bf16.json`)**:
   Exports `inclusionAI/Ling-3.0-tiny` in BF16 with paged attention.
2. **Block-FP8 (`inclusionAI-Ling-3.0-tiny_cuda_fp8.json`)**:
   Exports `inclusionAI/Ling-3.0-tiny-fp8` using 128x128 block-scaled `MatMulBlockQuantizedFp8Weight` and `QMoE(quant_type="fp8", block_size=128, swiglu_fusion=2)`.
3. **Quantized INT4 + INT8 KV (`inclusionAI-Ling-3.0-tiny_cuda_q4_int8kv.json`)**:
   Exports `inclusionAI/Ling-3.0-tiny` with INT4 `QMoE` (`block_size=64`), mixed INT4/INT8 symmetric `MatMulNBits` (`last_matmul:int8,mixed_layers:int8,linear_attn:int8`), INT8 embeddings (`GatherBlockQuantized(bits=8)`), and calibrated INT8 per-channel KV cache (`kv_scales_int8_per_channel.json`).
4. **Quantized INT8 + INT8 KV (`inclusionAI-Ling-3.0-tiny_cuda_q8_int8kv.json`)**:
   Exports `inclusionAI/Ling-3.0-tiny` with INT8 `QMoE` (`block_size=64`), INT8 symmetric `MatMulNBits`, INT8 embeddings (`GatherBlockQuantized(bits=8)`), and calibrated INT8 per-channel KV cache (`kv_scales_int8_per_channel.json`).

## Export

### Source prerequisites

These recipes require the source capabilities below, not just released `onnxruntime`, `onnxruntime-genai`, and Olive packages. The dependency pins are:

| Project | Revision | Required capability |
| --- | --- | --- |
| ONNX Runtime | `d93f5446f362d9c70bea81de953f32a26aa26d25` (merged [PR #33817](https://github.com/microsoft/onnxruntime/pull/33817)) or a descendant | Per-key KDA decay, `MoEGroupRouter`, paged INT8 KV, and FP8 operators |
| ONNX Runtime GenAI | `db92848ad9ed638fd9d2d691303e65f265341d75` (merged [PR #2898](https://github.com/microsoft/onnxruntime-genai/pull/2898)) or a descendant | Bailing loader/export support from [PR #2843](https://github.com/microsoft/onnxruntime-genai/pull/2843), native INT8 embeddings, and explicit routed-expert width |
| Olive | `ea6ae28111df93a82a0b04197751a57adba9d258` | ModelBuilder pre-checks and typed extra-option forwarding |

Olive at this revision still replaces GenAI's embedding exporter in its generic `ModelBuilder` pass. All four recipes therefore explicitly use the local `LingModelBuilder` registered in `package_config.json`. This subclass retains the native GenAI checkpoint, MatMul, and embedding exporters instead of applying patches for Olive-quantized checkpoints. Omitting the package configuration fails explicitly rather than silently exporting INT4 or floating-point embeddings.

The BF16 recipe requests `precision="bf16"` explicitly. The INT8 recipe sets `target_options.quant_config.io_dtype="bf16"`; `use_cuda_bf16` is a legacy INT4 option and does not select BF16 for INT8 or FP16 exports. The INT4 recipe retains that legacy option. The FP8 recipe is unchanged and is not covered by the precision regression tests below.

Use `transformers==5.15.0` with this checkpoint. Later strict configuration validation rejects its integer `kda_lower_bound`; do not patch the downloaded checkpoint to bypass that validation.

Full Ling exports still need Olive-side metadata fixes at the pinned revision: saved-config reloading does not forward `trust_remote_code`, and re-saving the checkpoint's generation metadata rejects inactive sampling hints. These are separate Olive issues, not reasons to change the GenAI model builder or its runtime sampling defaults.

On Linux, install a C++20 toolchain, CMake, CUDA 12.8 or newer and cuDNN 9, then build in a dedicated environment. FP8 QMoE requires a Hopper/SM90 GPU such as H100/H200 and **`onnxruntime_USE_FP8_QMOE=ON`**; a default ORT CUDA build is insufficient.

```bash
export SOURCE_ROOT="$PWD/ling-source"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDNN_HOME=/path/to/cudnn
mkdir -p "$SOURCE_ROOT"
python3 -m venv "$SOURCE_ROOT/venv"
source "$SOURCE_ROOT/venv/bin/activate"
python -m pip install numpy torch transformers==5.15.0 onnx onnx-ir safetensors huggingface-hub setuptools wheel packaging pytest

git clone https://github.com/microsoft/Olive.git "$SOURCE_ROOT/Olive"
git -C "$SOURCE_ROOT/Olive" checkout ea6ae28111df93a82a0b04197751a57adba9d258
python -m pip install -e "$SOURCE_ROOT/Olive"

git clone --recursive https://github.com/microsoft/onnxruntime.git "$SOURCE_ROOT/onnxruntime"
git -C "$SOURCE_ROOT/onnxruntime" checkout d93f5446f362d9c70bea81de953f32a26aa26d25
git -C "$SOURCE_ROOT/onnxruntime" submodule update --init --recursive
(
  cd "$SOURCE_ROOT/onnxruntime"
  ./build.sh --config Release --parallel --build_shared_lib --build_wheel \
    --use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" --skip_tests \
    --cmake_extra_defines onnxruntime_USE_FP8_QMOE=ON CMAKE_CUDA_ARCHITECTURES=90
)
export ORT_HOME="$SOURCE_ROOT/ort-install"
cmake --install "$SOURCE_ROOT/onnxruntime/build/Linux/Release" --config Release --prefix "$ORT_HOME"
# Keep only the custom CUDA ORT distribution in this dedicated environment.
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install "$SOURCE_ROOT"/onnxruntime/build/Linux/Release/dist/*.whl

git clone --recursive https://github.com/microsoft/onnxruntime-genai.git "$SOURCE_ROOT/onnxruntime-genai"
git -C "$SOURCE_ROOT/onnxruntime-genai" switch --detach db92848ad9ed638fd9d2d691303e65f265341d75
git -C "$SOURCE_ROOT/onnxruntime-genai" submodule update --init --recursive
python -m pip install -r "$SOURCE_ROOT/onnxruntime-genai/tools/python/requirements-core-packaging.txt"
(
  cd "$SOURCE_ROOT/onnxruntime-genai"
  python build.py --config Release --use_cuda --cuda_home "$CUDA_HOME" \
    --ort_home "$ORT_HOME" --parallel --update --build --package --skip_tests
)
python -m pip install "$SOURCE_ROOT"/onnxruntime-genai/build/Linux/Release/wheel/*.whl
export LD_LIBRARY_PATH="$ORT_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For BF16/INT4/INT8 on an A100, use `CMAKE_CUDA_ARCHITECTURES=80`; that does not enable FP8 QMoE execution on Ampere. Record `git rev-parse HEAD` in all three source directories alongside any validation results.

Run from this directory so that the custom pass and calibration file are resolved:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" olive run \
  --config inclusionAI-Ling-3.0-tiny_cuda_q4_int8kv.json \
  --package-config package_config.json --disable_telemetry
```

Use the same `--package-config` option for the other three recipes. Quantized exports must contain `/model/embed_tokens/GatherBlockQuantized` with `bits=8`, `gather_axis=0`, and `quantize_axis=1`; do not infer embedding precision from the recipe options alone.

### Validation status

`python -m pytest -q test_ling_model_builder.py` runs the BF16 and both quantized recipes through the actual Olive CLI with a tiny local dense checkpoint. It checks BF16 rather than FP16 floating-point initializers, INT8 embedding nodes, and INT4/INT8 MatMul bit widths. It does not download Ling weights, exercise Ling's custom configuration/metadata, or measure Ling accuracy. GenAI's `test/python/models/test_bailing_builder.py` separately covers Bailing scheduling, grouped router weights, KDA outputs/state, and both RoPE layouts against small numerical references.

The precision checks use the unmodified GenAI builder at `db92848ad9ed638fd9d2d691303e65f265341d75`, unmodified Olive at `ea6ae28111df93a82a0b04197751a57adba9d258`, and Transformers 5.15.0. They run offline without loading models on a GPU and do not require a GenAI generation-metadata patch.

Full Ling BF16/FP8/INT4/INT8 exports, generation accuracy versus SGLang/HF, and H200 latency/throughput validation are still required before approving these recipes. Include the dependency SHAs, GPU/CUDA versions, export commands, generation prompts/tokens, error tolerances, and performance measurements in that validation report. PR #689 should not be treated as ready merely because the offline checks pass.
