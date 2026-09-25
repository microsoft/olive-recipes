#!/bin/bash

olive capture-onnx-graph                                                   \
  --model_name_or_path openai/gpt-oss-20b                                  \
  --trust_remote_code                                                      \
  --execution_provider CUDAExecutionProvider                               \
  --precision int4                                                         \
  --use_model_builder                                                      \
  --use_ort_genai                                                          \
  --extra_mb_options op_types_to_quantize=MatMul/Gather                    \
                     moe_quant_type=int4                                   \
                     qmoe_fc1_type=int2                                    \
                     qmoe_fc2_type=int4                                    \
                     qmoe_block_size=64                                    \
  -o int4_cuda_int2_int4_qmoe