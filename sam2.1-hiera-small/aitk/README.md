# SAM2 Model Conversion

This repository demonstrates the optimization of the [sam2.1-hiera-small](https://github.com/facebookresearch/sam2) model using **post-training quantization (PTQ)** techniques.


### Generate ONNX Model
Activate the **Quantization Python Environment** and run command to generate encoder and decoder models:

```bash
python generate_model.py
```

### Run the Quantization + Compilation Config
Activate the **Quantization Python Environment** and run the workflow:

For Encoder Model:
```bash
olive run --config sam21_vision_encoder_qnn.json
```

For Decoder Model:
```bash
olive run --config sam21_mask_decoder_qnn.json
```

### Model ORT Execution

Execute SAM model in **AOT Compilation Python Environment** using following command:

```bash
python sam2_mask_generator.py --model_ve path/to/encoder_model.onnx --model_md path/to/decoder_model.onnx --image_path car.png --box_x 40 --box_y 235 --box_w 940 --box_h 490 --output_path car_mask.png
```
