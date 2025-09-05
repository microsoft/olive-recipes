#!/bin/bash

olive capture-onnx-graph                                        \
  --model_name_or_path openai/gpt-oss-20b                       \
  --trust_remote_code                                           \
  --conversion_device gpu                                       \
  --use_model_builder                                           \
  --use_ort_genai                                               \
  --extra_mb_options int4_op_types_to_quantize=MatMul/Gather    \
  -o int4_cuda_int4_qmoe
