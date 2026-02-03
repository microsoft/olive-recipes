## Whisper-large-v3-turbo Optimization with ONNX Runtime QNN EP
This folder outlines the process for optimizing the Whisper-large-v3-turbo model using ONNX Runtime with the QNN Execution Provider. It includes steps for exporting FP32 models, generating representative data for static quantization, creating QDQ models, model evaluation and performing audio transcription using the optimized models.

### Prerequisites
```bash
python -m pip install -r requirements_qnn.txt
```
### Generate data for static quantization

To get better results, we need to generate real data from original FP32 model instead of using random data for static quantization. Here we use 100 samples of librispeech dataset to generate the required real data which requires around 164 GB of disk space.

First generate FP32 onnx models:

1. Encoder FP32 model

    `olive run --config whisper_large_v3_turbo_encoder_fp32.json`
1. Decoder FP32 model

    `olive run --config whisper_large_v3_turbo_decoder_fp32.json`


Then download and generate data:

1. `python download_librispeech_asr.py --save_dir .\data`

2. `python .\demo.py --audio-path .\data\librispeech_asr_clean_test --encoder "models\whisper_encoder_fp32\model\model.onnx" --decoder "models\whisper_decoder_fp32\model.onnx" --model_id "openai/whisper-large-v3-turbo" --save_data .\data\quantization_data --num_data 100`

### Generate QDQ models

1. `olive run --config whisper_large_v3_turbo_encoder_qdq.json`
2. `olive run --config whisper_large_v3_turbo_decoder_qdq.json`

(Optional) Use whisper_large_v3_turbo_encoder_qdq_ctx.json and whisper_large_v3_turbo_decoder_qdq_ctx.json to create onnx models with QNN context binaries embedded in them.

### Evaluation

Evaluate model using the librispeech test-clean dataset

`python .\evaluate_whisper.py --encoder "models\whisper_encoder_qdq\model.onnx" --decoder "models\whisper_decoder_qdq\model.onnx" --model_id "openai/whisper-large-v3-turbo" --execution_provider QNNExecutionProvider`

### To transcribe a single sample:

`python .\demo.py --audio-path .\data\librispeech_asr_clean_test\1320-122617-0000.npy --encoder "models\whisper_encoder_qdq\model.onnx" --decoder "models\whisper_decoder_qdq\model.onnx" --model_id "openai/whisper-large-v3-turbo" --execution_provider QNNExecutionProvider`
