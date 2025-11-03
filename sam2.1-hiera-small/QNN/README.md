# SAM2 Model Conversion

This repository demonstrates the optimization of the [sam2.1-hiera-small](https://github.com/facebookresearch/sam2) model using **post-training quantization (PTQ)** techniques.


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

Replace `/path/to/qnn/env/bin` in [sam21_vision_encoder_qnn_ctx.json](sam21_vision_encoder_qnn_ctx.json) and [sam21_mask_decoder_qnn_ctx.json](sam21_mask_decoder_qnn_ctx.json) with the path to the directory containing your QNN environment's Python executable. This path can be found by running the following command in the environment:

```bash
# Linux
command -v python
# Windows
# where python
```

This command will return the path to the Python executable. Set the parent directory of the executable as the `/path/to/qnn/env/bin` in the config file.

### Generate ONNX Model
Activate the **Quantization Python Environment** and run command to generate encoder and decoder models:

```bash
python generate_model.py
```

### Run the Quantization + Compilation Config
Activate the **Quantization Python Environment** and run the workflow:

For Encoder Model:
```bash
olive run --config sam21_vision_encoder_qnn_ctx.json
```

For Decoder Model:
```bash
olive run --config sam21_mask_decoder_qnn_ctx.json
```

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.

### Model ORT Execution

Execute SAM model in **AOT Compilation Python Environment** using following command:

```bash
python sam2_mask_generator.py --model_ve path/to/encoder_model.onnx --model_md path/to/decoder_model.onnx --image_path car.png --box_x 40 --box_y 235 --box_w 940 --box_h 490 --output_path car_mask.png
```
