# Qwen3-30B-A3B Mixed INT2/INT4 RTN

This CUDA recipe exports Qwen3-30B-A3B with Olive RTN weight-only
quantization and Mobius FP16 graph construction. It reduces the gate/up
expert projections to INT2 in 24 of the model's 48 layers, while keeping
all attention projections and expert down projections at INT4.

## Quantization Layout

Layer indices are zero-based. The INT2 gate/up layers are:

```text
6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22,
24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40
```

| Component | Selected 24 layers | Other 24 layers |
|---|---|---|
| Expert gate/up (FC1/FC3) | INT2 | INT4 |
| Expert down (FC2) | INT4 | INT4 |
| Attention Q/K/V/O | INT4 | INT4 |

Token embeddings are INT4 and the language-model head is INT8.
All quantized components use symmetric quantization with group/block size
**128**, including the INT2 overrides. Floating-point tensors and activations
are exported at FP16. The `(2,4)` shorthand denotes gate/up versus down;
the actual QMoE FC1/FC2/FC3 attribute tuple is `(2,4,2)`.

## Export

Run from `Qwen-Qwen3-30B-A3B/` in a dedicated environment:

```bash
pip install -r cuda/rtn_mixed_int2_fp16/requirements.txt
olive run --config cuda/rtn_mixed_int2_fp16/config.json
```

The output directory is `cuda/rtn_mixed_int2_fp16/models/`.
The requirements pin the clean Olive and Mobius source revisions used for
the measured export. The model revision is
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

CUDA inference requires an ONNX Runtime build supporting INT2 MatMulNBits
and mixed-width INT2/INT4 QMoE, with compatible ONNX Runtime GenAI libraries.
The unpinned `onnxruntime-genai-cuda` dependency alone does not guarantee
that support. Install a compatible custom runtime after the requirements
when necessary. Runtime execution and accuracy have not been validated
for this export; the measurements below do not establish inference support.

## Model Size Comparison

Measured on October 4, 2026, using actual exported file lengths. Both exports
use FP16 graph precision, symmetric quantization, INT4 attention/embeddings,
and an INT8 head. Both exports use **group size 128**. The quantization
configuration differs only in the selected expert gate/up bit widths.
To reproduce the baseline, remove the gate/up override from this config,
retain the INT8 head override, and choose a separate `output_dir`.

| Setting | Mixed INT2/INT4 | INT4 Baseline |
|---|---|---|
| Selected expert gate/up weights | INT2 | INT4 |
| Other expert weights | INT4 | INT4 |
| Global group/block size | 128 | 128 |
| Attention / embeddings / head | INT4 / INT4 / INT8 | INT4 / INT4 / INT8 |

| Files | Mixed INT2/INT4 | INT4 Baseline |
|---|---:|---:|
| `model.onnx` | 902,958 bytes | 900,146 bytes |
| `model.onnx.data` | 13,512,409,088 bytes | 15,928,328,192 bytes |
| ONNX + external weights | 13,513,312,046 bytes | 15,929,228,338 bytes |
| ONNX + external weights (GB) | 13.5133 | 15.9292 |
| ONNX + external weights (GiB) | 12.5853 | 14.8352 |
| Entire export directory (bytes) | 13,524,758,664 | 15,940,674,010 |
| Entire export directory (GB) | 13.5248 | 15.9407 |

GB is decimal (`10^9` bytes); GiB is binary (`2^30` bytes). Directory totals
include tokenizer and pipeline metadata and can vary between runs.

The mixed model saves **2,415,916,292 bytes**, or **2.4159 GB / 2.2500 GiB**,
for the ONNX graph and external weights: a **15.17% reduction** versus
the block-128 INT4 baseline. These are disk sizes, not GPU memory measurements.

## Validation

The actual exported graph was checked for 48 QMoE nodes, the selected
24 `(2,4,2)` nodes and remaining 24 `(4,4,4)` nodes, block size 128,
192 INT4 attention MatMulNBits nodes (including all 48 V projections),
and an INT8 head. External weight file bounds and GenAI config JSON were
also checked. No inference, accuracy, or performance results are claimed.

Run the configuration schedule test with:

```bash
pip install pytest
python -m pytest cuda/rtn_mixed_int2_fp16/test_config.py -q
```