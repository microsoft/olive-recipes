# BGE-Small-EN-v1.5 Optimization
 
This folder contains examples of BGE-Small-EN-v1.5 optimization using different workflows for various hardware accelerators.
 
## Model Overview
 
BGE-Small-EN-v1.5 is a lightweight English text embedding model developed by BAAI (Beijing Academy of Artificial Intelligence). The model is optimized for sentence and text embedding tasks, providing high-quality vector representations for downstream applications such as semantic search, text classification, and similarity matching.
 
## Optimization Workflows
 
This directory provides three different optimization workflows targeting specific hardware accelerators:
 
- **QDQ for Qualcomm NPU**: Quantization-aware training for Qualcomm Neural Processing Units
- **QDQ for AMD NPU**: Quantization-aware training for AMD Neural Processing Units  
- **OpenVINO for Intel NPU**: OpenVINO optimization for Intel Neural Processing Units
 
## Workflow Details
 
### QDQ for Qualcomm NPU
 
This workflow performs quantization-aware training optimization for Qualcomm NPU acceleration. It follows the optimization pipeline:
 
- *HuggingFace Model → ONNX Model → Quantized ONNX Model*
 
**Configuration File**: `bge-small-en-v1.5_qdq_qnn.json`
 
**Key Features**:
- Uses QNN (Qualcomm Neural Network) execution provider
- Implements quantization-aware training with dynamic quantization
- Optimized for Qualcomm NPU hardware architecture
- Supports both activation and weight quantization
 
### QDQ for AMD NPU
 
This workflow performs quantization-aware training optimization for AMD NPU acceleration. It follows the optimization pipeline:
 
- *HuggingFace Model → ONNX Model → Quantized ONNX Model*
 
**Configuration File**: `bge-small-en-v1.5_qdq_amd.json`
 
**Key Features**:
- Optimized for AMD NPU architecture
- Implements quantization-aware training with dynamic quantization
- Enhanced performance for AMD hardware
- Supports both activation and weight quantization
 
### OpenVINO for Intel NPU
 
This workflow performs OpenVINO optimization for Intel NPU acceleration. It follows the optimization pipeline:
 
- *HuggingFace Model → OpenVINO IR Model*
 
**Configuration File**: `bge-small-en-v1.5_context_ov_static.json`
 
**Key Features**:
- Uses OpenVINO execution provider for Intel NPU
- Implements static quantization for optimal performance
- Custom user script for specialized data processing
- Enhanced accuracy evaluation using MTEB benchmarks
 
## Dataset Information
 
### Quantization Datasets
- **QNN/AMD NPU**: Uses MTEB Banking77 test split for quantization calibration
- **Intel NPU**: Uses Wikipedia train split (300 samples) with custom preprocessing
 
### Evaluation Datasets
- **Primary**: MTEB Banking77 classification task
- **Evaluation Metric**: Custom embedding accuracy for semantic similarity
- **Benchmark**: MTEB (Massive Text Embedding Benchmark) for standardized evaluation
 
## Performance Evaluation Results
 
The following results are based on comprehensive evaluation using standard embedding benchmarks and performance metrics. All evaluations use the MTEB Banking77 dataset for consistency.
 
### Qualcomm NPU (QNN) Performance
 
| Metric | Value |
|--------|-------|
| **Accuracy** | 85.57% |
| **Latency (avg)** | 14.83 ms |
| **Latency (min)** | 13.66 ms |
| **Latency (max)** | 17.92 ms |
| **Latency (p90)** | 15.52 ms |
| **Throughput (avg)** | 70.97 tokens/sec |
| **Throughput (max)** | 72.83 tokens/sec |
| **Throughput (min)** | 68.47 tokens/sec |
 
### AMD NPU Performance
 
| Metric | Value |
|--------|-------|
| **Accuracy** | 83.66% |
| **Latency (avg)** | 8.58 ms |
| **Latency (min)** | 7.54 ms |
| **Latency (max)** | 9.43 ms |
| **Latency (p90)** | 9.13 ms |
| **Throughput (avg)** | 107.26 tokens/sec |
| **Throughput (max)** | 130.15 tokens/sec |
| **Throughput (min)** | 88.90 tokens/sec |
 
### Intel NPU Performance
 
| Metric | Value |
|--------|-------|
| **Accuracy** | 85.42% |
| **Latency (avg)** | 3.33 ms |
| **Latency (min)** | 2.30 ms |
| **Latency (max)** | 6.39 ms |
| **Latency (p90)** | 4.01 ms |
| **Throughput (avg)** | 312.15 tokens/sec |
| **Throughput (max)** | 421.12 tokens/sec |
| **Throughput (min)** | 199.13 tokens/sec |
 
## Optimization Techniques
 
### Quantization Strategies
- **Dynamic Quantization**: Used for QNN and AMD NPU workflows
- **Static Quantization**: Used for Intel NPU workflow with OpenVINO
- **Mixed Precision**: Combines different precision levels for optimal performance
 
### Model Optimization Features
- **Input Optimization**: Fixed input shapes for better inference performance
- **Memory Optimization**: Efficient memory usage through quantization
- **Hardware-Specific Tuning**: Custom optimizations for each NPU architecture
 
## Requirements
 
The following dependencies are required for running the optimization workflows:
 
```
olive-ai
datasets
optimum
mteb
polars-lts-cpu (QNN only)
```
 
## Usage
 
1. **Select Workflow**: Choose the appropriate configuration file based on your target hardware:
   - For Qualcomm NPU: `bge-small-en-v1.5_qdq_qnn.json`
   - For AMD NPU: `bge-small-en-v1.5_qdq_amd.json`
   - For Intel NPU: `bge-small-en-v1.5_context_ov_static.json`
 
2. **Configure Parameters**: Adjust quantization parameters such as activation type, weight type, and quantization dataset according to your specific requirements.
 
3. **Run Optimization**: Execute the optimization pipeline using the selected configuration.
 
4. **Evaluate Results**: Use the provided evaluation scripts to assess model performance on your target hardware.
 
## Performance Notes
 
- **Accuracy**: Measured using custom embedding accuracy metrics from MTEB benchmark
- **Latency**: Measured in milliseconds per inference
- **Throughput**: Measured in tokens per second
- 
## Model Information
 
- **Model ID**: `BAAI/bge-small-en-v1.5`
- **Model Type**: Text Embedding Model
- **Framework**: HuggingFace Transformers
- **Optimization Target**: Hardware-specific acceleration for embedding generation
 
*Note: Performance metrics may vary depending on hardware specifications, system environment, and workload characteristics. The values provided here are for reference and may not reflect performance on all devices or configurations.*