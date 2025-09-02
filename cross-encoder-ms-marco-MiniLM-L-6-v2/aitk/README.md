# Cross-Encoder (MS MARCO MiniLM) optimization

This folder contains examples of optimization for `cross-encoder/ms-marco-MiniLM-L-6-v2` across multiple runtimes.

- QDQ for AMD NPU
- QDQ for Qualcomm NPU
- OpenVINO for Intel NPU
- Float downcasting for NVIDIA TRT (RTX) / DML for general GPU

## QDQ for AMD NPU

Quantize and package the model for AMD NPU.

- Pipeline: *HuggingFace Model -> ONNX -> Quantized ONNX*  
- Configuration File: `ms-marco-MiniLM-L-6-v2_qdq_amd.json` ([aitk/ms-marco-MiniLM-L-6-v2_qdq_amd.json](cross-encoder-ms-marco-MiniLM-L-6-v2/aitk/ms-marco-MiniLM-L-6-v2_qdq_amd.json))

Key features:
- Produce VitisAI-ready artifacts for AMD NPU deployment.  
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (pairwise scoring); inputs padded to 128 tokens.  
- Calibration/evaluation with XNLI (en) "premise" (≤10 samples).  
- ONNX export + transformer optimizations; static quantization to reduce latency.

## QDQ for Qualcomm NPU

Quantize the model for Qualcomm NPU runtimes.

- Pipeline: *HuggingFace Model -> ONNX -> Quantized ONNX*  
- Configuration File: `ms-marco-MiniLM-L-6-v2_qdq_qnn.json` ([aitk/ms-marco-MiniLM-L-6-v2_qdq_qnn.json](cross-encoder-ms-marco-MiniLM-L-6-v2/aitk/ms-marco-MiniLM-L-6-v2_qdq_qnn.json))

Key features:
- Target: Qualcomm NPU (QNN).  
- Calibration/evaluation uses XNLI (en) "premise" (≤10 samples, pad to 128).  
- ONNX export (opset 20), fix dynamic shapes to [1,128], apply light graph fusions, then static quantization (activations/weights tuned for QNN).

## OpenVINO (Intel NPU)

Convert and quantize the model for OpenVINO.

- Pipeline: *HuggingFace Model -> ONNX -> OpenVINO quantized model*  
- Configuration File: `ms-marco-MiniLM-L-6-v2_context_ov_static.json` ([aitk/ms-marco-MiniLM-L-6-v2_context_ov_static.json](cross-encoder-ms-marco-MiniLM-L-6-v2/aitk/ms-marco-MiniLM-L-6-v2_context_ov_static.json))

Key features:
- OpenVINO Optimum conversion and encapsulation for Intel NPU.  
- Calibration uses Wikipedia (train, 300 samples) with a custom transform for transformer inputs.  
- Enforces static I/O (three inputs of [1,128]) and applies transformer-specific quantization.

## Float downcasting for NVIDIA TRT / DML

FP16 export for GPU backends to improve throughput.

- Pipeline: *HuggingFace Model -> ONNX -> FP16 ONNX*  
- Configuration Files:  
  - TRT (RTX): `ms-marco-MiniLM-L-6-v2_trtrtx.json` ([aitk/ms-marco-MiniLM-L-6-v2_trtrtx.json](cross-encoder-ms-marco-MiniLM-L-6-v2/aitk/ms-marco-MiniLM-L-6-v2_trtrtx.json))  
  - DML: `ms-marco-MiniLM-L-6-v2_dml.json` ([aitk/ms-marco-MiniLM-L-6-v2_dml.json](cross-encoder-ms-marco-MiniLM-L-6-v2/aitk/ms-marco-MiniLM-L-6-v2_dml.json))

Key features:
- FP16 conversion for TensorRT and DML-optimized exports for GPU inference.  
- Fixed-shape [1,128] inputs and transformer optimizations for stable latency/throughput.  
- Evaluate latency and throughput on XNLI example inputs.

## Dataset Information

### Quantization Datasets
- **QNN/AMD NPU**: XNLI (English validation) "premise" (≤10 samples, padded to 128).  
- **Intel NPU (OpenVINO)**: Wikipedia train (300 samples) with transformer-specific preprocessing.

### Evaluation Datasets
- **Primary**: XNLI (English) validation for quick latency/throughput checks.  
- **Metric**: Pairwise scoring (ranking) metrics or embedding/latency metrics depending on evaluation script.  
- **Benchmark**: Use task-appropriate benchmarks (MS MARCO for ranking, XNLI for latency/throughput examples).

### Evaluation Datasets
- **Primary**: XNLI (English) validation split for NLI classification.  
- **Evaluation Metric**: Embedding-based classification accuracy (or feature-extraction latency/throughput).  
- **Benchmark**: XNLI (Cross-lingual Natural Language Inference).

## Performance Evaluation Results
The following results are based on comprehensive evaluation using standard embedding benchmarks and performance metrics. 

### Qualcomm NPU (QNN) Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 5.59 ms |
| **Latency (min)** | 4.87 ms |
| **Latency (max)** | 7.13 ms |
| **Throughput (avg)** | 190.41 tokens/sec |
| **Throughput (max)** | 206.31 tokens/sec |
| **Throughput (min)** | 138.28 tokens/sec |

### AMD NPU Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 5.78 ms |
| **Latency (min)** | 4.92 ms |
| **Latency (max)** | 7.77 ms |
| **Throughput (avg)** | 186.78 tokens/sec |
| **Throughput (max)** | 229.40 tokens/sec |
| **Throughput (min)** | 137.03 tokens/sec |

### Intel NPU Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 2.59 ms |
| **Latency (p90)** | 3.26 ms |
| **Similarity** | 0.9830 |

### TRT Performance

| Metric | Value |
|--------|-------|
| **Latency (avg)** | 0.63 ms |
| **Latency (min)** | 0.59 ms |
| **Latency (max)** | 0.72 ms |
| **Throughput (avg)** | 780.33 tokens/sec |
| **Throughput (max)** | 1557.39 tokens/sec |
| **Throughput (min)** | 171.56 tokens/sec |

### DML Performance

| Metric | Value |
|--------|-------|
| **Latency (max)** | 2.02 ms |
| **Latency (min)** | 1.34 ms |
| **Throughput (avg)** | 684.74 tokens/sec |
| **Throughput (max)** | 721.03 tokens/sec |
| **Throughput (min)** | 560.07 tokens/sec |

## Notes
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (pairwise cross-encoder for ranking).  
- Use the listed config files to reproduce conversion, quantization, and benchmark runs.