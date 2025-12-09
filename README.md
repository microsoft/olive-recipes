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
| bert | clip | deepseek | hiera | llama | llama3 | mistral | mobilenet | phi3 | phi4 | qwen2 | resnet | vit |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/QNN) | [OFA-Sys-chinese-clip-vit-base-patch16](OFA-Sys-chinese-clip-vit-base-patch16/aitk) | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/aitk) | [sam2.1-hiera-small](sam2.1-hiera-small/QNN) | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/NvTensorRtRtx) | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk) | [mistralai-Mistral-7B-Instruct-v0.2](mistralai-Mistral-7B-Instruct-v0.2/NvTensorRtRtx) | [timm-mobilenetv3_small_100.lamb_in1k](timm-mobilenetv3_small_100.lamb_in1k/VitisAI) | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/NvTensorRtRtx) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/aitk) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/NvTensorRtRtx) | [microsoft-resnet-50](microsoft-resnet-50/aitk) | [google-vit-base-patch16-224](google-vit-base-patch16-224/OpenVINO) |
| [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/QNN) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/QNN) |  | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/NvTensorRtRtx) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/QNN) | [mistralai-Mistral-7B-Instruct-v0.2](mistralai-Mistral-7B-Instruct-v0.2/aitk) |  | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/QNN) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/olive) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk) |  | [google-vit-base-patch16-224](google-vit-base-patch16-224/QNN) |
| [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/QNN) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk) |  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/NvTensorRtRtx) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk) | [mistralai-Mistral-7B-Instruct-v0.3](mistralai-Mistral-7B-Instruct-v0.3/aitk) |  | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/aitk) | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk) | [Qwen-Qwen2.5-0.5B](Qwen-Qwen2.5-0.5B/aitk) |  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk) |
| [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/QNN) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/olive) |  |  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/olive) |  |  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/NvTensorRtRtx) | [microsoft-Phi-4-reasoning-plus](microsoft-Phi-4-reasoning-plus/aitk) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/NvTensorRtRtx) |  | [sam-vit-base](sam-vit-base/QNN) |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/aitk) |  |  |  |  |  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/QNN) | [microsoft-Phi-4-reasoning](microsoft-Phi-4-reasoning/aitk) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/QNN) |  |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/QNN) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk) |  |  |  |  |  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk) | [microsoft-Phi-4](microsoft-Phi-4/OpenVINO) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk) |  |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk) |  |  |  |  |  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/NvTensorRtRtx) | [microsoft-Phi-4](microsoft-Phi-4/aitk) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/olive) |  |  |
|  | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk) |  |  |  |  |  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN) |  | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/NvTensorRtRtx) |  |  |
|  |  |  |  |  |  |  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk) |  | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/olive) |  | [Qwen-Qwen2.5-3B-Instruct](Qwen-Qwen2.5-3B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  | [microsoft-Phi-4](microsoft-Phi-4/NvTensorRtRtx) |  | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/NvTensorRtRtx) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/NvTensorRtRtx) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/NvTensorRtRtx) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/NvTensorRtRtx) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-3B-Instruct](Qwen-Qwen2.5-Coder-3B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/NvTensorRtRtx) |  |  |
|  |  |  |  |  |  |  |  |  |  | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk) |  |  |
|  |  |  |  |  |  |  |  |  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/NvTensorRtRtx) |  |  |
|  |  |  |  |  |  |  |  |  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/NvTensorRtRtx) |  |  |
|  |  |  |  |  |  |  |  |  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/NvTensorRtRtx) |  |  |
<!-- end_arch_models -->
</details>

<details>
<summary>Models grouped by device</summary></br>

<!-- begin_device_models -->
| cpu | gpu | npu |
| :---: | :---: | :---: |
| [OFA-Sys-chinese-clip-vit-base-patch16](OFA-Sys-chinese-clip-vit-base-patch16/aitk/openai_clip_ov.json) | [DeepSeek-R1-Distill-Llama-8B_Model_Builder_INT4](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/NvTensorRtRtx/DeepSeek-R1-Distill-Llama-8B_model_builder_int4.json) | [OFA-Sys-chinese-clip-vit-base-patch16](OFA-Sys-chinese-clip-vit-base-patch16/aitk/openai_clip_ov.json) |
| [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_gpu_config.json) | [DeepSeek-R1-Distill-Qwen-1.5B_Model_Builder_FP16](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/NvTensorRtRtx/DeepSeek-R1-Distill-Qwen-1.5B_model_builder_fp16.json) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk/qwen2_5_ov_npu_config.json) |
| [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_gpu_config.json) | [DeepSeek-R1-Distill-Qwen-14B_NVMO_INT4_AWQ](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/NvTensorRtRtx/DeepSeek-R1-Distill-Qwen-14B_nvmo_int4_awq.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/QNN/config.json) |
| [gemma-3-1b-it_model_builder_cpu_FP32](google-gemma/olive/gemma-3-1b-it_model_builder_cpu_fp32.json) | [DeepSeek-R1-Distill-Qwen-7B_NVMO_INT4_RTN](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/NvTensorRtRtx/DeepSeek-R1-Distill-Qwen-7B_nvmo_int4_rtn.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_config.json) |
| [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_context_ov_static.json) | [Llama-3.2-1B-Instruct_Model_Builder_FP16](meta-llama-Llama-3.2-1B-Instruct/NvTensorRtRtx/Llama-3.2-1B-Instruct_model_builder_fp16.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_qnn_config.json) |
| [google-gemma](google-gemma/olive/README.md) | [Llama3.1-8B-Instruct_Model_Builder_INT4](meta-llama-Llama-3.1-8B-Instruct/NvTensorRtRtx/Llama-3.1-8B-Instruct_model_builder_int4.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_vitis_ai_config.json) |
| [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit_base_patch16_224_context_ov_static.json) | [Mistral-7B-Instruct-v0.2_Model_Builder_INT4](mistralai-Mistral-7B-Instruct-v0.2/NvTensorRtRtx/Mistral-7B-Instruct-v0.2_model_builder_int4.json) | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk/qwen2_5_ov_npu_config.json) |
| [intel-bert-base-uncased-mrpc (ov)](intel-bert-base-uncased-mrpc/aitk/bert_ov.json) | [OFA-Sys-chinese-clip-vit-base-patch16](OFA-Sys-chinese-clip-vit-base-patch16/aitk/openai_clip_ov.json) | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk/qwen2_5_ov_npu_config.json) |
| [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_ov.json) | [Phi-3-mini-128k-instruct_NVMO_INT4_RTN](microsoft-Phi-3-mini-128k-instruct/NvTensorRtRtx/Phi-3-mini-128k-instruct_nvmo_int4_rtn.json) | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk/qwen2_5_ov_npu_config.json) |
| [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_ov_gpu_config.json) | [Phi-3-mini-4k-instruct_Model_Builder_INT4](microsoft-Phi-3-mini-4k-instruct/NvTensorRtRtx/Phi-3-mini-4k-instruct_model_builder_int4.json) | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk/qwen2_5_ov_npu_config.json) |
| [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_gpu_config.json) | [Phi3.5_Mini_Instruct_Model_Builder_INT4](microsoft-Phi-3.5-mini-instruct/NvTensorRtRtx/Phi-3.5-mini-instruct_model_builder_int4.json) | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/aitk/deepseek_ov_npu_config.json) |
| [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_gpu_config.json) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk/qwen2_5_ov_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_config.json) |
| [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_context_ov_static.json) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk/qwen2_5_trtrtx.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_qnn_config.json) |
| [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_ov.json) | [Qwen-Qwen2.5-0.5B](Qwen-Qwen2.5-0.5B/aitk/qwen2_5_ov_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_vitis_ai_config.json) |
| [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_ov.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/QNN/config_gpu.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk/deepseek_ov_npu_config.json) |
| [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_ov.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/QNN/config_gpu_ctxbin.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_context_ov_static.json) |
| [timm-mobilenetv3_small_100.lamb_in1k](timm-mobilenetv3_small_100.lamb_in1k/olive/config.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_dml_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_qdq_amd.json) |
|  | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_gpu_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_qdq_qnn.json) |
|  | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_trtrtx_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/OpenVINO/vit_base_patch16_224_context_ov_static.json) |
|  | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/aitk/qwen2_5_ov_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/QNN/vit_qnn_fp32_ctx.json) |
|  | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/aitk/qwen2_5_trtrtx.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_qdq_amd.json) |
|  | [Qwen-Qwen2.5-3B-Instruct](Qwen-Qwen2.5-3B-Instruct/aitk/qwen2_5_ov_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_qdq_qnn.json) |
|  | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk/qwen2_5_ov_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit_base_patch16_224_context_ov_static.json) |
|  | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk/qwen2_5_trtrtx.json) | [intel-bert-base-uncased-mrpc (AMD)](intel-bert-base-uncased-mrpc/aitk/bert_qdq_amd.json) |
|  | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk/qwen2_5_ov_config.json) | [intel-bert-base-uncased-mrpc (ov)](intel-bert-base-uncased-mrpc/aitk/bert_ov.json) |
|  | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk/qwen2_5_trtrtx.json) | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_qdq_qnn.json) |
|  | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk/qwen2_5_ov_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_ov.json) |
|  | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk/qwen2_5_trtrtx.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_qdq_amd.json) |
|  | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/aitk/qwen2_5_ov_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_qnn.json) |
|  | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/aitk/qwen2_5_trtrtx.json) | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_ov_config.json) |
|  | [Qwen-Qwen2.5-Coder-3B-Instruct](Qwen-Qwen2.5-Coder-3B-Instruct/aitk/qwen2_5_ov_config.json) | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_qnn_config.json) |
|  | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk/qwen2_5_ov_config.json) | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_vitis_ai_config.json) |
|  | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk/qwen2_5_trtrtx.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_config.json) |
|  | [Qwen2.5-0.5B-Instruct_Model_Builder_FP16](Qwen-Qwen2.5-0.5B-Instruct/NvTensorRtRtx/Qwen2.5-0.5B-Instruct_model_builder_fp16.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_qnn_config.json) |
|  | [Qwen2.5-14B-Instruct_Model_Builder_INT4](Qwen-Qwen2.5-14B-Instruct/NvTensorRtRtx/Qwen2.5-14B-Instruct_model_builder_int4.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_vitis_ai_config.json) |
|  | [Qwen2.5-7B-Instruct_Model_Builder_INT4](Qwen-Qwen2.5-7B-Instruct/NvTensorRtRtx/Qwen2.5-7B-Instruct_model_builder_int4.json) | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/QNN/config.json) |
|  | [Qwen2.5-Coder-0.5B-Instruct_Model_Builder_FP16](Qwen-Qwen2.5-Coder-0.5B-Instruct/NvTensorRtRtx/Qwen2.5-Coder-0.5B-Instruct_model_builder_fp16.json) | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/QNN/config.json) |
|  | [Qwen2.5-Coder-1.5B-Instruct_Model_Builder_FP16](Qwen-Qwen2.5-Coder-1.5B-Instruct/NvTensorRtRtx/Qwen2.5-Coder-1.5B-Instruct_model_builder_fp16.json) | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk/phi3_ov_npu_config.json) |
|  | [Qwen2.5-Coder-14B-Instruct_Model_Builder_INT4](Qwen-Qwen2.5-Coder-14B-Instruct/NvTensorRtRtx/Qwen2.5-Coder-14B-Instruct_model_builder_int4.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN/config.json) |
|  | [Qwen2.5-Coder-7B-Instruct_Model_Builder_INT4](Qwen-Qwen2.5-Coder-7B-Instruct/NvTensorRtRtx/Qwen2.5-Coder-7B-Instruct_model_builder_int4.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN/config_fp16.json) |
|  | [Qwen2.5_1.5B_Instruct_Model_Builder_FP16](Qwen-Qwen2.5-1.5B-Instruct/NvTensorRtRtx/Qwen2.5-1.5B-Instruct_model_builder_fp16.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_config.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/aitk/deepseek_ov_config.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_qnn_config.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/QNN/config_gpu.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_vitis_ai_config.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/QNN/config_gpu_ctxbin.json) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/QNN/phi4_mini_qnn_docker.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_dml_config.json) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/aitk/phi4_ov_npu_config.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_gpu_config.json) | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk/phi4_ov_config.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_trtrtx_config.json) | [microsoft-Phi-4-reasoning-plus](microsoft-Phi-4-reasoning-plus/aitk/phi4_ov_config.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/aitk/deepseek_ov_config.json) | [microsoft-Phi-4-reasoning](microsoft-Phi-4-reasoning/aitk/phi4_ov_config.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/aitk/deepseek_trtrtx.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_context_ov_static.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk/deepseek_ov_config.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_qdq_amd.json) |
|  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk/deepseek_trtrtx.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_qdq_qnn.json) |
|  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/QNN/config_gpu_fp32.json) | [microsoft-table-transformer-detection](microsoft-table-transformer-detection/QNN/ttd_config.json) |
|  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_context_ov_static.json) | [mistralai-Mistral-7B-Instruct-v0.2](mistralai-Mistral-7B-Instruct-v0.2/aitk/Mistral_7B_Instruct_v0.2_npu_context_ov_dy.json) |
|  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_dml.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_ov.json) |
|  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_migraphx.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_qdq_amd.json) |
|  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_trtrtx.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_qnn.json) |
|  | [google-gemma](google-gemma/olive/README.md) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_ov.json) |
|  | [google-vit-base-patch16-224](google-vit-base-patch16-224/QNN/config_gpu_fp32.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_qdq_amd.json) |
|  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_dml.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_qnn.json) |
|  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_migraphx.json) | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_ov.json) |
|  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_trtrtx.json) | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_qdq_amd.json) |
|  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit_base_patch16_224_context_ov_static.json) | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_qnn.json) |
|  | [intel-bert-base-uncased-mrpc (ov)](intel-bert-base-uncased-mrpc/aitk/bert_ov.json) | [sam-vit-base](sam-vit-base/QNN/sam_mask_box_decoder_qnn_fp16.json) |
|  | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/QNN/config_gpu_fp32.json) | [sam-vit-base](sam-vit-base/QNN/sam_mask_decoder_qnn_fp16_ctx.json) |
|  | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_dml.json) | [sam-vit-base](sam-vit-base/QNN/sam_mask_point_decoder_qnn_fp16.json) |
|  | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_migraphx.json) | [sam-vit-base](sam-vit-base/QNN/sam_vision_encoder_qnn_w8a8_ctx.json) |
|  | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_trtrtx.json) | [sam2.1-hiera-small](sam2.1-hiera-small/QNN/sam21_mask_decoder_qnn_ctx.json) |
|  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/QNN/config_gpu_fp32.json) | [sam2.1-hiera-small](sam2.1-hiera-small/QNN/sam21_vision_encoder_qnn_ctx.json) |
|  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_dml.json) | [timm-mobilenetv3_small_100.lamb_in1k](timm-mobilenetv3_small_100.lamb_in1k/QNN/mobilenet_qnn_ep.json) |
|  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_migraphx.json) | [timm-mobilenetv3_small_100.lamb_in1k](timm-mobilenetv3_small_100.lamb_in1k/VitisAI/config.json) |
|  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_ov.json) |  |
|  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_trtrtx.json) |  |
|  | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_dml_config.json) |  |
|  | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_ov_gpu_config.json) |  |
|  | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_trtrtx_config.json) |  |
|  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/QNN/config_gpu.json) |  |
|  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/QNN/config_gpu_ctxbin.json) |  |
|  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_dml_config.json) |  |
|  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_gpu_config.json) |  |
|  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_trtrtx_config.json) |  |
|  | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/aitk/phi3_ov_config.json) |  |
|  | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/aitk/phi3_trtrtx.json) |  |
|  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk/phi3_ov_config.json) |  |
|  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk/phi3_trtrtx.json) |  |
|  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN/config_gpu.json) |  |
|  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN/config_gpu_ctxbin.json) |  |
|  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_dml_config.json) |  |
|  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_gpu_config.json) |  |
|  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_trtrtx_config.json) |  |
|  | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/OpenVINO/Phi-4-mini-instruct-gpu-context-dy.json) |  |
|  | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/OpenVINO/Phi_4_mini_instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |
|  | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/aitk/phi4_ov_config.json) |  |
|  | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/OpenVINO/Phi-4-mini-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |
|  | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/OpenVINO/Phi_4_mini_instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |
|  | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk/phi4_ov_gpu_config.json) |  |
|  | [microsoft-Phi-4-reasoning-plus](microsoft-Phi-4-reasoning-plus/OpenVINO/Phi-4-Phi-4-reasoning-plus_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |
|  | [microsoft-Phi-4-reasoning](microsoft-Phi-4-reasoning/OpenVINO/Phi-4-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |
|  | [microsoft-Phi-4](microsoft-Phi-4/OpenVINO/phi_4_gpu_context_dy.json) |  |
|  | [microsoft-Phi-4](microsoft-Phi-4/aitk/phi4_ov_config.json) |  |
|  | [microsoft-Phi-4](microsoft-Phi-4/aitk/phi4_trtrtx.json) |  |
|  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_context_ov_static.json) |  |
|  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_dml.json) |  |
|  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_migraphx.json) |  |
|  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_trtrtx.json) |  |
|  | [mistralai-Mistral-7B-Instruct-v0.2](mistralai-Mistral-7B-Instruct-v0.2/aitk/Mistral_7B_Instruct_v0.2_gpu_context_ov_dy.json) |  |
|  | [mistralai-Mistral-7B-Instruct-v0.2](mistralai-Mistral-7B-Instruct-v0.2/aitk/Mistral_7B_Instruct_v0.2_trtrtx.json) |  |
|  | [mistralai-Mistral-7B-Instruct-v0.3](mistralai-Mistral-7B-Instruct-v0.3/aitk/mistral-7b-instruct-v0.3-ov.json) |  |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/QNN/config_gpu_fp32.json) |  |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_dml.json) |  |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_migraphx.json) |  |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_ov.json) |  |
|  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_trtrtx.json) |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/QNN/config_gpu_fp32.json) |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_dml.json) |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_migraphx.json) |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_ov.json) |  |
|  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_trtrtx.json) |  |
|  | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_dml.json) |  |
|  | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_migraphx.json) |  |
|  | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_ov.json) |  |
|  | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_trtrtx.json) |  |
|  | [phi-4_Model_Builder_INT4](microsoft-Phi-4/NvTensorRtRtx/phi-4_model_builder_int4.json) |  |
<!-- end_device_models -->
</details>

<details>
<summary>Models grouped by EP</summary></br>

<!-- begin_ep_models -->
| CPU | CUDA | Dml | MIGraphX | NvTensorRTRTX | OpenVINO | QNN | VitisAI |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [gemma-3-1b-it_model_builder_cpu_FP32](google-gemma/olive/gemma-3-1b-it_model_builder_cpu_fp32.json) | [google-gemma](google-gemma/olive/README.md) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_dml_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_migraphx.json) | [DeepSeek-R1-Distill-Llama-8B_Model_Builder_INT4](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/NvTensorRtRtx/DeepSeek-R1-Distill-Llama-8B_model_builder_int4.json) | [OFA-Sys-chinese-clip-vit-base-patch16](OFA-Sys-chinese-clip-vit-base-patch16/aitk/openai_clip_ov.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/QNN/config.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_vitis_ai_config.json) |
| [google-gemma](google-gemma/olive/README.md) |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_dml_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_migraphx.json) | [DeepSeek-R1-Distill-Qwen-1.5B_Model_Builder_FP16](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/NvTensorRtRtx/DeepSeek-R1-Distill-Qwen-1.5B_model_builder_fp16.json) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk/qwen2_5_ov_config.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/QNN/config_gpu.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_vitis_ai_config.json) |
| [timm-mobilenetv3_small_100.lamb_in1k](timm-mobilenetv3_small_100.lamb_in1k/olive/config.json) |  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_dml.json) | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_migraphx.json) | [DeepSeek-R1-Distill-Qwen-14B_NVMO_INT4_AWQ](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/NvTensorRtRtx/DeepSeek-R1-Distill-Qwen-14B_nvmo_int4_awq.json) | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk/qwen2_5_ov_npu_config.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/QNN/config_gpu_ctxbin.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_qdq_amd.json) |
|  |  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_dml.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_migraphx.json) | [DeepSeek-R1-Distill-Qwen-7B_NVMO_INT4_RTN](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/NvTensorRtRtx/DeepSeek-R1-Distill-Qwen-7B_nvmo_int4_rtn.json) | [Qwen-Qwen2.5-0.5B](Qwen-Qwen2.5-0.5B/aitk/qwen2_5_ov_config.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_qnn_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_qdq_amd.json) |
|  |  | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_dml.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_migraphx.json) | [Llama-3.2-1B-Instruct_Model_Builder_FP16](meta-llama-Llama-3.2-1B-Instruct/NvTensorRtRtx/Llama-3.2-1B-Instruct_model_builder_fp16.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/QNN/config_gpu.json) | [intel-bert-base-uncased-mrpc (AMD)](intel-bert-base-uncased-mrpc/aitk/bert_qdq_amd.json) |
|  |  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_dml.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_migraphx.json) | [Llama3.1-8B-Instruct_Model_Builder_INT4](meta-llama-Llama-3.1-8B-Instruct/NvTensorRtRtx/Llama-3.1-8B-Instruct_model_builder_int4.json) | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_ov_gpu_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/QNN/config_gpu_ctxbin.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_qdq_amd.json) |
|  |  | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_dml_config.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_migraphx.json) | [Mistral-7B-Instruct-v0.2_Model_Builder_INT4](mistralai-Mistral-7B-Instruct-v0.2/NvTensorRtRtx/Mistral-7B-Instruct-v0.2_model_builder_int4.json) | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/aitk/qwen2_5_ov_config.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_qnn_config.json) | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_vitis_ai_config.json) |
|  |  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_dml_config.json) | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_migraphx.json) | [Phi-3-mini-128k-instruct_NVMO_INT4_RTN](microsoft-Phi-3-mini-128k-instruct/NvTensorRtRtx/Phi-3-mini-128k-instruct_nvmo_int4_rtn.json) | [Qwen-Qwen2.5-3B-Instruct](Qwen-Qwen2.5-3B-Instruct/aitk/qwen2_5_ov_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/QNN/config_gpu_fp32.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_vitis_ai_config.json) |
|  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_dml_config.json) |  | [Phi-3-mini-4k-instruct_Model_Builder_INT4](microsoft-Phi-3-mini-4k-instruct/NvTensorRtRtx/Phi-3-mini-4k-instruct_model_builder_int4.json) | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk/qwen2_5_ov_config.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_qdq_qnn.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_vitis_ai_config.json) |
|  |  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_dml.json) |  | [Phi3.5_Mini_Instruct_Model_Builder_INT4](microsoft-Phi-3.5-mini-instruct/NvTensorRtRtx/Phi-3.5-mini-instruct_model_builder_int4.json) | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk/qwen2_5_ov_npu_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/QNN/config_gpu_fp32.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_qdq_amd.json) |
|  |  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_dml.json) |  | [Qwen-Qwen2.5-0.5B-Instruct](Qwen-Qwen2.5-0.5B-Instruct/aitk/qwen2_5_trtrtx.json) | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk/qwen2_5_ov_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/QNN/vit_qnn_fp32_ctx.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_qdq_amd.json) |
|  |  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_dml.json) |  | [Qwen-Qwen2.5-1.5B-Instruct](Qwen-Qwen2.5-1.5B-Instruct/aitk/qwen2_5_trtrtx_config.json) | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk/qwen2_5_ov_npu_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_qdq_qnn.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_qdq_amd.json) |
|  |  | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_dml.json) |  | [Qwen-Qwen2.5-14B-Instruct](Qwen-Qwen2.5-14B-Instruct/aitk/qwen2_5_trtrtx.json) | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk/qwen2_5_ov_config.json) | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/QNN/config_gpu_fp32.json) | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_qdq_amd.json) |
|  |  |  |  | [Qwen-Qwen2.5-7B-Instruct](Qwen-Qwen2.5-7B-Instruct/aitk/qwen2_5_trtrtx.json) | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk/qwen2_5_ov_npu_config.json) | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_qdq_qnn.json) | [timm-mobilenetv3_small_100.lamb_in1k](timm-mobilenetv3_small_100.lamb_in1k/VitisAI/config.json) |
|  |  |  |  | [Qwen-Qwen2.5-Coder-0.5B-Instruct](Qwen-Qwen2.5-Coder-0.5B-Instruct/aitk/qwen2_5_trtrtx.json) | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/aitk/qwen2_5_ov_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/QNN/config_gpu_fp32.json) |  |
|  |  |  |  | [Qwen-Qwen2.5-Coder-1.5B-Instruct](Qwen-Qwen2.5-Coder-1.5B-Instruct/aitk/qwen2_5_trtrtx.json) | [Qwen-Qwen2.5-Coder-3B-Instruct](Qwen-Qwen2.5-Coder-3B-Instruct/aitk/qwen2_5_ov_config.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_qnn.json) |  |
|  |  |  |  | [Qwen-Qwen2.5-Coder-14B-Instruct](Qwen-Qwen2.5-Coder-14B-Instruct/aitk/qwen2_5_trtrtx.json) | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk/qwen2_5_ov_config.json) | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_qnn_config.json) |  |
|  |  |  |  | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk/qwen2_5_trtrtx.json) | [Qwen-Qwen2.5-Coder-7B-Instruct](Qwen-Qwen2.5-Coder-7B-Instruct/aitk/qwen2_5_ov_npu_config.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/QNN/config_gpu.json) |  |
|  |  |  |  | [Qwen2.5-0.5B-Instruct_Model_Builder_FP16](Qwen-Qwen2.5-0.5B-Instruct/NvTensorRtRtx/Qwen2.5-0.5B-Instruct_model_builder_fp16.json) | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/aitk/deepseek_ov_config.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/QNN/config_gpu_ctxbin.json) |  |
|  |  |  |  | [Qwen2.5-14B-Instruct_Model_Builder_INT4](Qwen-Qwen2.5-14B-Instruct/NvTensorRtRtx/Qwen2.5-14B-Instruct_model_builder_int4.json) | [deepseek-ai-DeepSeek-R1-Distill-Llama-8B](deepseek-ai-DeepSeek-R1-Distill-Llama-8B/aitk/deepseek_ov_npu_config.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_qnn_config.json) |  |
|  |  |  |  | [Qwen2.5-7B-Instruct_Model_Builder_INT4](Qwen-Qwen2.5-7B-Instruct/NvTensorRtRtx/Qwen2.5-7B-Instruct_model_builder_int4.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_config.json) | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/QNN/config.json) |  |
|  |  |  |  | [Qwen2.5-Coder-0.5B-Instruct_Model_Builder_FP16](Qwen-Qwen2.5-Coder-0.5B-Instruct/NvTensorRtRtx/Qwen2.5-Coder-0.5B-Instruct_model_builder_fp16.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_ov_gpu_config.json) | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/QNN/config.json) |  |
|  |  |  |  | [Qwen2.5-Coder-1.5B-Instruct_Model_Builder_FP16](Qwen-Qwen2.5-Coder-1.5B-Instruct/NvTensorRtRtx/Qwen2.5-Coder-1.5B-Instruct_model_builder_fp16.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/aitk/deepseek_ov_config.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN/config.json) |  |
|  |  |  |  | [Qwen2.5-Coder-14B-Instruct_Model_Builder_INT4](Qwen-Qwen2.5-Coder-14B-Instruct/NvTensorRtRtx/Qwen2.5-Coder-14B-Instruct_model_builder_int4.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk/deepseek_ov_config.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN/config_fp16.json) |  |
|  |  |  |  | [Qwen2.5-Coder-7B-Instruct_Model_Builder_INT4](Qwen-Qwen2.5-Coder-7B-Instruct/NvTensorRtRtx/Qwen2.5-Coder-7B-Instruct_model_builder_int4.json) | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk/deepseek_ov_npu_config.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN/config_gpu.json) |  |
|  |  |  |  | [Qwen2.5_1.5B_Instruct_Model_Builder_FP16](Qwen-Qwen2.5-1.5B-Instruct/NvTensorRtRtx/Qwen2.5-1.5B-Instruct_model_builder_fp16.json) | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_context_ov_static.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/QNN/config_gpu_ctxbin.json) |  |
|  |  |  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B](deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_trtrtx_config.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/OpenVINO/vit_base_patch16_224_context_ov_static.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_qnn_config.json) |  |
|  |  |  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-14B](deepseek-ai-DeepSeek-R1-Distill-Qwen-14B/aitk/deepseek_trtrtx.json) | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit_base_patch16_224_context_ov_static.json) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/QNN/phi4_mini_qnn_docker.json) |  |
|  |  |  |  | [deepseek-ai-DeepSeek-R1-Distill-Qwen-7B](deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/aitk/deepseek_trtrtx.json) | [intel-bert-base-uncased-mrpc (ov)](intel-bert-base-uncased-mrpc/aitk/bert_ov.json) | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_qdq_qnn.json) |  |
|  |  |  |  | [google-bert-bert-base-multilingual-cased](google-bert-bert-base-multilingual-cased/aitk/bert-base-multilingual-cased_trtrtx.json) | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_ov.json) | [microsoft-table-transformer-detection](microsoft-table-transformer-detection/QNN/ttd_config.json) |  |
|  |  |  |  | [google-vit-base-patch16-224](google-vit-base-patch16-224/aitk/vit-base-patch16-224_trtrtx.json) | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_ov_config.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/QNN/config_gpu_fp32.json) |  |
|  |  |  |  | [intel-bert-base-uncased-mrpc](intel-bert-base-uncased-mrpc/aitk/bert_trtrtx.json) | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_ov_gpu_config.json) | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_qnn.json) |  |
|  |  |  |  | [laion-CLIP-ViT-B-32-laion2B-s34B-b79K](laion-CLIP-ViT-B-32-laion2B-s34B-b79K/aitk/laion_clip_trtrtx.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_config.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/QNN/config_gpu_fp32.json) |  |
|  |  |  |  | [meta-llama-Llama-3.1-8B-Instruct](meta-llama-Llama-3.1-8B-Instruct/aitk/llama3_1_trtrtx_config.json) | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_ov_gpu_config.json) | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_qnn.json) |  |
|  |  |  |  | [meta-llama-Llama-3.2-1B-Instruct](meta-llama-Llama-3.2-1B-Instruct/aitk/llama3_2_trtrtx_config.json) | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/aitk/phi3_ov_config.json) | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_qnn.json) |  |
|  |  |  |  | [microsoft-Phi-3-mini-128k-instruct](microsoft-Phi-3-mini-128k-instruct/aitk/phi3_trtrtx.json) | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk/phi3_ov_config.json) | [sam-vit-base](sam-vit-base/QNN/sam_mask_box_decoder_qnn_fp16.json) |  |
|  |  |  |  | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk/phi3_trtrtx.json) | [microsoft-Phi-3-mini-4k-instruct](microsoft-Phi-3-mini-4k-instruct/aitk/phi3_ov_npu_config.json) | [sam-vit-base](sam-vit-base/QNN/sam_mask_decoder_qnn_fp16_ctx.json) |  |
|  |  |  |  | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_trtrtx_config.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_config.json) | [sam-vit-base](sam-vit-base/QNN/sam_mask_point_decoder_qnn_fp16.json) |  |
|  |  |  |  | [microsoft-Phi-4](microsoft-Phi-4/aitk/phi4_trtrtx.json) | [microsoft-Phi-3.5-mini-instruct](microsoft-Phi-3.5-mini-instruct/aitk/phi3_5_ov_gpu_config.json) | [sam-vit-base](sam-vit-base/QNN/sam_vision_encoder_qnn_w8a8_ctx.json) |  |
|  |  |  |  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_trtrtx.json) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/OpenVINO/Phi-4-mini-instruct-gpu-context-dy.json) | [sam2.1-hiera-small](sam2.1-hiera-small/QNN/sam21_mask_decoder_qnn_ctx.json) |  |
|  |  |  |  | [mistralai-Mistral-7B-Instruct-v0.2](mistralai-Mistral-7B-Instruct-v0.2/aitk/Mistral_7B_Instruct_v0.2_trtrtx.json) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/OpenVINO/Phi_4_mini_instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) | [sam2.1-hiera-small](sam2.1-hiera-small/QNN/sam21_vision_encoder_qnn_ctx.json) |  |
|  |  |  |  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_trtrtx.json) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/aitk/phi4_ov_config.json) | [timm-mobilenetv3_small_100.lamb_in1k](timm-mobilenetv3_small_100.lamb_in1k/QNN/mobilenet_qnn_ep.json) |  |
|  |  |  |  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_trtrtx.json) | [microsoft-Phi-4-mini-instruct](microsoft-Phi-4-mini-instruct/aitk/phi4_ov_npu_config.json) |  |  |
|  |  |  |  | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_trtrtx.json) | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/OpenVINO/Phi-4-mini-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |  |
|  |  |  |  | [phi-4_Model_Builder_INT4](microsoft-Phi-4/NvTensorRtRtx/phi-4_model_builder_int4.json) | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/OpenVINO/Phi_4_mini_instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |  |
|  |  |  |  |  | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk/phi4_ov_config.json) |  |  |
|  |  |  |  |  | [microsoft-Phi-4-mini-reasoning](microsoft-Phi-4-mini-reasoning/aitk/phi4_ov_gpu_config.json) |  |  |
|  |  |  |  |  | [microsoft-Phi-4-reasoning-plus](microsoft-Phi-4-reasoning-plus/OpenVINO/Phi-4-Phi-4-reasoning-plus_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |  |
|  |  |  |  |  | [microsoft-Phi-4-reasoning-plus](microsoft-Phi-4-reasoning-plus/aitk/phi4_ov_config.json) |  |  |
|  |  |  |  |  | [microsoft-Phi-4-reasoning](microsoft-Phi-4-reasoning/OpenVINO/Phi-4-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) |  |  |
|  |  |  |  |  | [microsoft-Phi-4-reasoning](microsoft-Phi-4-reasoning/aitk/phi4_ov_config.json) |  |  |
|  |  |  |  |  | [microsoft-Phi-4](microsoft-Phi-4/OpenVINO/phi_4_gpu_context_dy.json) |  |  |
|  |  |  |  |  | [microsoft-Phi-4](microsoft-Phi-4/aitk/phi4_ov_config.json) |  |  |
|  |  |  |  |  | [microsoft-resnet-50](microsoft-resnet-50/aitk/resnet_context_ov_static.json) |  |  |
|  |  |  |  |  | [mistralai-Mistral-7B-Instruct-v0.2](mistralai-Mistral-7B-Instruct-v0.2/aitk/Mistral_7B_Instruct_v0.2_gpu_context_ov_dy.json) |  |  |
|  |  |  |  |  | [mistralai-Mistral-7B-Instruct-v0.2](mistralai-Mistral-7B-Instruct-v0.2/aitk/Mistral_7B_Instruct_v0.2_npu_context_ov_dy.json) |  |  |
|  |  |  |  |  | [mistralai-Mistral-7B-Instruct-v0.3](mistralai-Mistral-7B-Instruct-v0.3/aitk/mistral-7b-instruct-v0.3-ov.json) |  |  |
|  |  |  |  |  | [openai-clip-vit-base-patch16](openai-clip-vit-base-patch16/aitk/openai_clip_ov.json) |  |  |
|  |  |  |  |  | [openai-clip-vit-base-patch32](openai-clip-vit-base-patch32/aitk/openai_clip_ov.json) |  |  |
|  |  |  |  |  | [openai-clip-vit-large-patch14](openai-clip-vit-large-patch14/aitk/openai_clip_ov.json) |  |  |
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
