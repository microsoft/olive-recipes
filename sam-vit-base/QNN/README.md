# SAM Model Conversion

This repository demonstrates the optimization of the [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base) model using **post-training quantization (PTQ)** techniques.


### Quantization Python Environment Setup
Quantization is resource-intensive and requires GPU acceleration

For GPU Environment Setup:
```bash
pip install -r requirements.txt
```

### AOT Compilation Python Environment Setup
Model compilation using QNN Execution Provider requires a Python environment with onnxruntime-qnn installed. In a separate Python environment, install the required packages:

```bash
# Install Olive & ORT-QNN
pip install olive-ai onnxruntime-qnn torch torchvision transformers
```

Replace `/path/to/qnn/env/bin` in [sam_vision_encoder_qnn_w8a8_ctx.json](sam_vision_encoder_qnn_w8a8_ctx.json) and [sam_mask_decoder_qnn_fp16_ctx.json](sam_mask_decoder_qnn_fp16_ctx.json) with the path to the directory containing your QNN environment's Python executable. This path can be found by running the following command in the environment:

```bash
# Linux
command -v python
# Windows
# where python
```

This command will return the path to the Python executable. Set the parent directory of the executable as the `/path/to/qnn/env/bin` in the config file.

### Run the Quantization + Compilation Config
Activate the **Quantization Python Environment** and run the workflow:

For Encoder Model:
```bash
olive run --config sam_vision_encoder_qnn_w8a8_ctx.json
```

For Decoder Model provides two types of inputs, Point and Bounbary Box. Based on requirement use one of the following workflows.

For Point based Decoder Model:
```bash
olive run --config sam_mask_point_decoder_qnn_fp16.json
```
For Box based Decoder Model:
```bash
olive run --config sam_mask_box_decoder_qnn_fp16.json
```
For Point and Box based Decoder Model:
```bash
olive run --config sam_mask_decoder_qnn_fp16_ctx.json
```

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.

### Model ORT Execution

Execute SAM model in **AOT Compilation Python Environment** using following command:

```bash
python sam_mask_generator.py --model_ve path/to/encoder_model.onnx --model_md path/to/decoder_model.onnx --image_path car.png --box_x 40 --box_y 235 --box_w 940 --box_h 490 --output_path car_mask.png
```
