<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".assets/olive-white-text.png">
    <source media="(prefers-color-scheme: light)" srcset=".assets/olive-black-text.png">
    <img alt="olive" src=".assets/olive-black-text.png" height="100" style="max-width: 100%;">
  </picture>

## Olive Recipes For AI Model Optimization Toolkit
</div>

This repository compliments [Olive](https://github.com/microsoft/Olive), the AI model optimization toolkit, and includes recipes demonstrating its extensive features and use cases. Users of Olive can use these recipes as a reference to either optimize publicly available AI models or to optimize their own proprietary models.

## Supported models, architectures, devices and execution providers
Below are list of available recipes grouped by different criteria. Click the link to expand.

<details>
<summary>Models grouped by model architecture</summary></br>

<!-- begin_arch_models -->
| bert | clip | deepseek | llama | llama3 | mistral | phi3 | phi4 | qwen2 | resnet | vit |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk) | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/aitk) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/NvTensorRtRtx) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk) | [mistralai-Mistral-7B-Instruct-v0.3](mistralai-Mistral-7B-Instruct-v0.3/aitk) | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/aitk) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/aitk) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk) | [microsoft-resnet-50](microsoft-resnet-50/aitk) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk) |
| [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk) |  |  |  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk) | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk) | [Qwen-Qwen2.5-0.5B](Qwen-Qwen2.5-0.5B/aitk) |  |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/aitk) |  |  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/NvTensorRtRtx) | [microsoft-Phi-4-reasoning-plus](microsoft-Phi-4-reasoning-plus/aitk) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/NvTensorRtRtx) |  |  |
|  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk) |  |  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk) | [microsoft-Phi-4-reasoning](microsoft-Phi-4-reasoning/aitk) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-3B-Instruct](Qwen-Qwen2.5-3B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-3B-Instruct](Qwen-Qwen2.5-Coder-3B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/NvTensorRtRtx) |  |  |
<!-- end_arch_models -->
</details>

<details>
<summary>Models grouped by device</summary></br>

<!-- begin_device_models -->
| cpu | gpu | npu |
| :---: | :---: | :---: |
| [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_config.json) | [DeepSeek_R1_1.5B_FP16_Model_Builder](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/NvTensorRtRtx/DeepSeek-R1-Distill-Qwen-1.5B_fp16_model_builder.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_config.json) |
| [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_config.json) | [Llama3.2_1B_Instruct_NVMO_INT4_AWQ](meta-llama-Llama-3.2-1B-Instruct/NvTensorRtRtx/Llama-3.2-1B-Instruct_nvmo_int4_awq.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_qnn_config.json) |
| [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_context_ov_static.json) | [Phi3.5_Mini_Instruct_NVMO_INT4_AWQ](microsoft-Phi-3.5-mini-instruct/NvTensorRtRtx/Phi-3.5-mini-instruct_nvmo_int4_awq.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_vitis_ai_config.json) |
| [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit_base_patch16_224_context_ov_static.json) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk/qwen2_5_ov_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_config.json) |
| [intel-bert-base-uncased-mrpc (ov)](intel-bert-base-uncased-mrpc/aitk/bert_ov.json) | [Qwen-Qwen2.5-0.5B](Qwen-Qwen2.5-0.5B/aitk/qwen2_5_ov_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_qnn_config.json) |
| [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_ov.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_dml_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_vitis_ai_config.json) |
| [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_config.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_context_ov_static.json) |
| [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_config.json) | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/aitk/qwen2_5_ov_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_qdq_amd.json) |
| [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk/phi4_ov_config.json) | [Qwen-Qwen2.5-3B-Instruct](Qwen-Qwen2.5-3B-Instruct/aitk/qwen2_5_ov_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_qdq_qnn.json) |
| [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_context_ov_static.json) | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk/qwen2_5_ov_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_qdq_amd.json) |
| [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_ov.json) | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk/qwen2_5_ov_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_qdq_qnn.json) |
| [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_ov.json) | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk/qwen2_5_ov_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit_base_patch16_224_context_ov_static.json) |
|  | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/aitk/qwen2_5_ov_config.json) | [intel-bert-base-uncased-mrpc (AMD)](intel-bert-base-uncased-mrpc/aitk/bert_qdq_amd.json) |
|  | [Qwen-Qwen2.5-Coder-3B-Instruct](Qwen-Qwen2.5-Coder-3B-Instruct/aitk/qwen2_5_ov_config.json) | [intel-bert-base-uncased-mrpc (ov)](intel-bert-base-uncased-mrpc/aitk/bert_ov.json) |
|  | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk/qwen2_5_ov_config.json) | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_qdq_qnn.json) |
|  | [Qwen2.5_1.5B_Instruct_NVMO_INT4_AWQ](Qwen-Qwen2.5-1.5B-Instruct/NvTensorRtRtx/Qwen2.5-1.5B-Instruct_nvmo_int4_awq.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K (Text)](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_text_qnn.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/aitk/deepseek_ov_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K (Vision)](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_vision_qnn.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_dml_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_ov.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_qdq_amd.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/aitk/deepseek_ov_config.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_config.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk/deepseek_ov_config.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_qnn_config.json) |
|  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_context_ov_static.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_vitis_ai_config.json) |
|  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_dml.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_config.json) |
|  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_trtrtx.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_qnn_config.json) |
|  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_dml.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_vitis_ai_config.json) |
|  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_trtrtx.json) | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk/phi4_ov_config.json) |
|  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit_base_patch16_224_context_ov_static.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_context_ov_static.json) |
|  | [intel-bert-base-uncased-mrpc (ov)](intel-bert-base-uncased-mrpc/aitk/bert_ov.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_qdq_amd.json) |
|  | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_dml.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_qdq_qnn.json) |
|  | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_trtrtx.json) | [openai-clip-vit-base-patch16 (Text)](openai-clip-vit-base-patch16/aitk/openai_clip_text_qnn.json) |
|  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_dml.json) | [openai-clip-vit-base-patch16 (Vision)](openai-clip-vit-base-patch16/aitk/openai_clip_vision_qnn.json) |
|  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_ov.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_ov.json) |
|  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_trtrtx.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_qdq_amd.json) |
|  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_dml_config.json) | [openai-clip-vit-base-patch32 (Text)](openai-clip-vit-base-patch32/aitk/openai_clip_text_qnn.json) |
|  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_config.json) | [openai-clip-vit-base-patch32 (Vision)](openai-clip-vit-base-patch32/aitk/openai_clip_vision_qnn.json) |
|  | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/aitk/phi3_ov_config.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_ov.json) |
|  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk/phi3_ov_config.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_qdq_amd.json) |
|  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_dml_config.json) |  |
|  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_config.json) |  |
|  | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/aitk/phi4_ov_config.json) |  |
|  | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk/phi4_ov_config.json) |  |
|  | [microsoft-Phi-4-reasoning-plus](microsoft-Phi-4-reasoning-plus/aitk/phi4_ov_config.json) |  |
|  | [microsoft-Phi-4-reasoning](microsoft-Phi-4-reasoning/aitk/phi4_ov_config.json) |  |
|  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_context_ov_static.json) |  |
|  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_dml.json) |  |
|  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_trtrtx.json) |  |
|  | [mistralai-Mistral-7B-Instruct-v0.3](mistralai-Mistral-7B-Instruct-v0.3/aitk/mistral-7b-instruct-v0.3-ov.json) |  |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_dml.json) |  |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_ov.json) |  |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_trtrtx.json) |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_dml.json) |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_ov.json) |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_trtrtx.json) |  |
<!-- end_device_models -->
</details>

<details>
<summary>Models grouped by EP</summary></br>

<!-- begin_ep_models -->
| Dml | NvTensorRTRTX | OpenVINO | QNN | VitisAI |
| :---: | :---: | :---: | :---: | :---: |
| [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_dml_config.json) | [DeepSeek_R1_1.5B_FP16_Model_Builder](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/NvTensorRtRtx/DeepSeek-R1-Distill-Qwen-1.5B_fp16_model_builder.json) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk/qwen2_5_ov_config.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_qnn_config.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_vitis_ai_config.json) |
| [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_dml_config.json) | [Llama3.2_1B_Instruct_NVMO_INT4_AWQ](meta-llama-Llama-3.2-1B-Instruct/NvTensorRtRtx/Llama-3.2-1B-Instruct_nvmo_int4_awq.json) | [Qwen-Qwen2.5-0.5B](Qwen-Qwen2.5-0.5B/aitk/qwen2_5_ov_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_qnn_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_vitis_ai_config.json) |
| [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_dml.json) | [Phi3.5_Mini_Instruct_NVMO_INT4_AWQ](microsoft-Phi-3.5-mini-instruct/NvTensorRtRtx/Phi-3.5-mini-instruct_nvmo_int4_awq.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_qdq_qnn.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_qdq_amd.json) |
| [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_dml.json) | [Qwen2.5_1.5B_Instruct_NVMO_INT4_AWQ](Qwen-Qwen2.5-1.5B-Instruct/NvTensorRtRtx/Qwen2.5-1.5B-Instruct_nvmo_int4_awq.json) | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/aitk/qwen2_5_ov_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_qdq_qnn.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_qdq_amd.json) |
| [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_dml.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_trtrtx.json) | [Qwen-Qwen2.5-3B-Instruct](Qwen-Qwen2.5-3B-Instruct/aitk/qwen2_5_ov_config.json) | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_qdq_qnn.json) | [intel-bert-base-uncased-mrpc (AMD)](intel-bert-base-uncased-mrpc/aitk/bert_qdq_amd.json) |
| [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_dml.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_trtrtx.json) | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk/qwen2_5_ov_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K (Text)](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_text_qnn.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_qdq_amd.json) |
| [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_dml_config.json) | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_trtrtx.json) | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk/qwen2_5_ov_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K (Vision)](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_vision_qnn.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_vitis_ai_config.json) |
| [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_dml_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_trtrtx.json) | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk/qwen2_5_ov_config.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_qnn_config.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_vitis_ai_config.json) |
| [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_dml.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_trtrtx.json) | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/aitk/qwen2_5_ov_config.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_qnn_config.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_qdq_amd.json) |
| [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_dml.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_trtrtx.json) | [Qwen-Qwen2.5-Coder-3B-Instruct](Qwen-Qwen2.5-Coder-3B-Instruct/aitk/qwen2_5_ov_config.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_qdq_qnn.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_qdq_amd.json) |
| [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_dml.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_trtrtx.json) | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk/qwen2_5_ov_config.json) | [openai-clip-vit-base-patch16 (Text)](openai-clip-vit-base-patch16/aitk/openai_clip_text_qnn.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_qdq_amd.json) |
|  |  | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/aitk/deepseek_ov_config.json) | [openai-clip-vit-base-patch16 (Vision)](openai-clip-vit-base-patch16/aitk/openai_clip_vision_qnn.json) |  |
|  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_config.json) | [openai-clip-vit-base-patch32 (Text)](openai-clip-vit-base-patch32/aitk/openai_clip_text_qnn.json) |  |
|  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/aitk/deepseek_ov_config.json) | [openai-clip-vit-base-patch32 (Vision)](openai-clip-vit-base-patch32/aitk/openai_clip_vision_qnn.json) |  |
|  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk/deepseek_ov_config.json) |  |  |
|  |  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_context_ov_static.json) |  |  |
|  |  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit_base_patch16_224_context_ov_static.json) |  |  |
|  |  | [intel-bert-base-uncased-mrpc (ov)](intel-bert-base-uncased-mrpc/aitk/bert_ov.json) |  |  |
|  |  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_ov.json) |  |  |
|  |  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_config.json) |  |  |
|  |  | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/aitk/phi3_ov_config.json) |  |  |
|  |  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk/phi3_ov_config.json) |  |  |
|  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_config.json) |  |  |
|  |  | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/aitk/phi4_ov_config.json) |  |  |
|  |  | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk/phi4_ov_config.json) |  |  |
|  |  | [microsoft-Phi-4-reasoning-plus](microsoft-Phi-4-reasoning-plus/aitk/phi4_ov_config.json) |  |  |
|  |  | [microsoft-Phi-4-reasoning](microsoft-Phi-4-reasoning/aitk/phi4_ov_config.json) |  |  |
|  |  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_context_ov_static.json) |  |  |
|  |  | [mistralai-Mistral-7B-Instruct-v0.3](mistralai-Mistral-7B-Instruct-v0.3/aitk/mistral-7b-instruct-v0.3-ov.json) |  |  |
|  |  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_ov.json) |  |  |
|  |  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_ov.json) |  |  |
<!-- end_ep_models -->
</details>

## Learn more
- [Olive Github Repository](https://github.com/microsoft/Olive)
- [Getting Started](https://github.com/microsoft/Olive#-getting-started)
- [Olive Documentation](https://microsoft.github.io/Olive)

## 🤝 Contributions and Feedback
- We welcome contributions! Please read the [contribution guidelines](./CONTRIBUTING.md) for more details on how to contribute to the Olive project.
- For feature requests or bug reports, file a [GitHub Issue](https://github.com/microsoft/Olive/issues).
- For general discussion or questions, use [GitHub Discussions](https://github.com/microsoft/Olive/discussions).

## ⚖️ License
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.