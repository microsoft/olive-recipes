# Export GPT-OSS-20B with Mixed-Width CUDA QMoE

This recipe exports the dense weights as INT4 and requantizes the GPT-OSS
MXFP4 experts to symmetric mixed-width QMoE weights:

- gate/up projections (FC1/FC3): INT2
- down projection (FC2): INT4
- QMoE block size: 64

The generated model targets the CUDA execution provider. ONNX Runtime performs
the final expert-weight prepacking when the model is loaded.

## Prerequisites

Install the latest Olive and ONNX Runtime GenAI CUDA packages:

```bash
python -m pip install -r ../requirements.txt
```

Until the changes are available in nightly packages, build from the branches in:

- [ONNX Runtime GenAI #2624](https://github.com/microsoft/onnxruntime-genai/pull/2624)
- [ONNX Runtime #32761](https://github.com/microsoft/onnxruntime/pull/32761)

## Export

```bash
./gpt-oss-20b.sh
```

The exported model is saved in `int4_cuda_int2_int4_qmoe`.

## Run

Use the ONNX Runtime GenAI sample chat application:

```bash
python model-chat.py -m int4_cuda_int2_int4_qmoe/model -e cuda
```