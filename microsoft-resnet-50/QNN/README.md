# ResNet Optimization with PTQ on Qualcomm NPU using QNN EP

This example performs ResNet optimization on Qualcomm NPU with ONNX Runtime PTQ. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

It requires x86 python environment on a Windows ARM machine with `onnxruntime-qnn` installed.

**NOTE:** The model quantization part of the workflow can also be done on a Linux/Windows machine with a different onnxruntime package installed. Remove the `"evaluators"` and `"evaluator"` sections from the configuration file to skip the evaluation step.

### QNN-GPU:

Please install Olive directly using:

```bash
pip install olive-ai
```

To run the config:

```bash
olive run --config config_gpu_fp32.json
```

✅ Optimized model saved in: `output/`
