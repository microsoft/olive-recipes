# TIMM Model Optimization (Quantization)

This folder contains examples of **TIMM (PyTorch Image Models) optimization** using **Olive workflows**, focusing on **ONNX quantization** with QuarkQuantization pass.

## **Optimization Workflow**

This example optimizes `timm/mobilenetv3_small_100.lamb_in1k` for **CPU or NPU execution** by:
- *Converting PyTorch model to ONNX*
- *Applying ONNX quantization*

- **Model**: [timm/mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **Dataset**: [timm/mini-imagenet](https://huggingface.co/datasets/timm/mini-imagenet)

---

## **Running the Optimization**

### **Running with Config File**

The provided `config.json` configuration performs **ONNX conversion and quantization**.

**Install Required Dependencies**

```sh
pip install -r requirements.txt
olive run --config config.json --setup
```

**Run Model Optimization**

```sh
olive run --config config.json
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
