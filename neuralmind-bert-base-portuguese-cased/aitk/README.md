# NeuralMind BERT optimization

This folder contains examples of BERT optimization for `neuralmind/bert-base-portuguese-cased` using different workflows.

- QDQ for AMD NPU
- QDQ for Qualcomm NPU
- OpenVINO for Intel NPU
- Float downcasting for NVIDIA TRT (RTX) / DML for general GPU

## QDQ for AMD NPU

This example quantizes the model via QDQ and targets AMD NPU.

- Pipeline: *HuggingFace Model -> Onnx Model -> Quantized Onnx Model*
- Configuration File: `bert-base-portuguese-cased_qdq_amd.json` 

Key features:
- Produces VitisAI-ready artifacts for AMD NPU.
- Uses `neuralmind/bert-base-portuguese-cased`; inputs padded to 128.
- Calibration on XNLI "premise" (≤10 samples).
- Transformer optimizations + static quantization (int8 weights, uint16 activations) for lower latency.


## QDQ for Qualcomm NPU

This example quantizes the model for Qualcomm NPU.

- Pipeline: *HuggingFace Model -> Onnx Model -> Quantized Onnx Model*
- Configuration File: `bert-base-portuguese-cased_qdq_qnn.json` 


**Key features**:
- Target: Qualcomm NPU (QNN).
- Calibrate/evaluate on XNLI "premise" (≤10 samples, pad 128).
- ONNX export, fix shapes to [1,128], then static quantization (uint16/uint8).

## OpenVINO (Intel NPU)

This example converts and quantizes via OpenVINO for Intel NPU.

- Pipeline: *HuggingFace Model -> Onnx Model -> Quantized Onnx Model*
- Configuration File: `bert-base-portuguese-cased_context_ov_static.json` 


**Key features**:
- Convert to OpenVINO and encapsulate for Intel NPU.
- Calibrate on Wikipedia (train, 300 samples) with a custom transform.
- Static I/O (three inputs of [1,128]) and transformer-specific quantization.

## Float downcasting for NVIDIA TRT / DML

Float16 downcast workflows for GPU backends.

- Pipeline: *HuggingFace Model -> Onnx Model -> Float16 Onnx Model*
- Configuration Files:  
  - TRT (RTX): `bert-base-portuguese-cased_trtrtx.json` 
  - DML: `bert-base-portuguese-cased_dml.json`

**Key features**:
- Convert to FP16 for TensorRT; GPU-optimized export for DML.
- Fixed inputs [1,128], transformer optimizations; evaluate latency and throughput.

## Dataset Information

### Quantization Datasets
- **QNN/AMD NPU**: XNLI (English validation) using the "premise" field (≤10 samples, padded to 128).  
- **Intel NPU (OpenVINO)**: Wikipedia train split, 300 samples (transformer-specific preprocessing).

### Evaluation Datasets
- **Primary**: XNLI (English) validation split for NLI classification.  
- **Evaluation Metric**: Embedding-based classification accuracy (or feature-extraction latency/throughput).  
- **Benchmark**: XNLI (Cross-lingual Natural Language Inference).

## Performance Evaluation Results
The following results are based on comprehensive evaluation using standard embedding benchmarks and performance metrics. 

### Qualcomm NPU (QNN) Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 14.71 ms |
| **Latency (min)** | 16.29 ms |
| **Latency (max)** | 13.98 ms |
| **Throughput (avg)** | 66.10 tokens/sec |
| **Throughput (max)** | 71.92 tokens/sec |
| **Throughput (min)** | 37.79 tokens/sec |

### AMD NPU Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 11.78 ms |
| **Latency (min)** | 12.24 ms |
| **Latency (max)** | 11.40 ms |
| **Throughput (avg)** | 82.40 tokens/sec |
| **Throughput (max)** | 87.26 tokens/sec |
| **Throughput (min)** | 79.16 tokens/sec |

### Intel NPU Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 4.36 ms |
| **Latency (p90)** | 4.95 ms |
| **Similarity** | 0.9845 |

### TRT Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 1.99 ms |
| **Latency (min)** | 1.34 ms |
| **Latency (max)** | 2.39 ms |
| **Throughput (avg)** | 597.05 tokens/sec |
| **Throughput (max)** | 741.56 tokens/sec |
| **Throughput (min)** | 191.27 tokens/sec |

### DML Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 3.69 ms |
| **Latency (min)** | 2.98 ms |
| **Throughput (avg)** | 268.21 tokens/sec |
| **Throughput (max)** | 330.60 tokens/sec |
| **Throughput (min)** | 89.13 tokens/sec |

## Notes
- Model used: `neuralmind/bert-base-portuguese-cased` (feature extraction).  
- Evaluations in the recipes focus on latency and throughput for feature-extraction tasks.  
- Run the corresponding recipe files to reproduce conversion, quantization, and evaluation workflows.