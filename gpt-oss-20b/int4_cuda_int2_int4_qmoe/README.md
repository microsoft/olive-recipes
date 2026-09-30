# Export GPT-OSS-20B with Mixed-Width CUDA QMoE

This recipe exports the dense weights as INT4 and requantizes the GPT-OSS
MXFP4 experts to symmetric mixed-width QMoE weights:

- gate/up projections (FC1/FC3): INT2
- down projection (FC2): INT4
- QMoE block size: 64

The generated model targets the CUDA execution provider. ONNX Runtime performs
the final expert-weight prepacking when the model is loaded.

The recipe uses the structured mixed-width QMoE configuration introduced by
[ONNX Runtime GenAI #2624](https://github.com/microsoft/onnxruntime-genai/pull/2624)
and the packed CUDA decode support introduced by
[ONNX Runtime #32761](https://github.com/microsoft/onnxruntime/pull/32761).

## Prerequisites

Install the latest Olive and ONNX Runtime GenAI CUDA packages:

```bash
python -m pip install -r ../requirements.txt
```

The `accelerate` and `kernels` packages in the shared requirements are needed
to load the official GPT-OSS MXFP4 checkpoint through Transformers during
export. They are not dependencies of the exported ONNX model.

Both changes have been merged. Use package versions that include them. Exporting
the official GPT-OSS checkpoint also requires an ONNX Runtime GenAI build that
loads expert tensors from the checkpoint's `model.layers.*.mlp.experts` keys.

## Export

Run the command from this recipe directory:

```bash
olive capture-onnx-graph \
	--model_name_or_path openai/gpt-oss-20b \
	--trust_remote_code \
	--execution_provider CUDAExecutionProvider \
	--precision int4 \
	--use_model_builder \
	--use_ort_genai \
	--extra_mb_options "builder_config_version=2,target_options=target-options.json" \
	-o int4_cuda_int2_int4_qmoe
```

The exported model is saved in `int4_cuda_int2_int4_qmoe`.

The resulting GPT-OSS-20B model contains 24 mixed-width QMoE nodes with INT2
gate/up projections, INT4 down projections, and block size 64.

## Runtime Status

The packed INT2/INT4 CUDA kernel currently targets decode workloads with at most
eight expanded rows. GPT-OSS routes each token to four experts, so this covers up
to two input tokens per QMoE invocation. Longer prefill inputs currently select
the dense dequantization fallback and can exceed its default scratch-memory
limit.

Model loading and token-by-token decode are supported, but the standard
`model-chat.py` application performs multi-token prefill and is not supported by
this recipe until ONNX Runtime provides a bounded mixed-width prefill path. Do
not raise `ep.cuda.qmoe_int_dequant_max_scratch_bytes` as a production
workaround because that fallback materializes the expert weights.

The model directory passed to ONNX Runtime GenAI is
`int4_cuda_int2_int4_qmoe` (the directory directly containing `model.onnx` and
`genai_config.json`).
