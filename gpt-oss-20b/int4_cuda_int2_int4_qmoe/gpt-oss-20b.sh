#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

olive capture-onnx-graph                                                   \
  --model_name_or_path openai/gpt-oss-20b                                  \
  --trust_remote_code                                                      \
  --execution_provider CUDAExecutionProvider                               \
  --precision int4                                                         \
  --use_model_builder                                                      \
  --use_ort_genai                                                          \
  --extra_mb_options "builder_config_version=2,target_options=${SCRIPT_DIR}/target-options.json" \
  -o int4_cuda_int2_int4_qmoe
