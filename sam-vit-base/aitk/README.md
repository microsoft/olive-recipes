# SAM Model Conversion

This repository demonstrates the optimization of the [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base) model using **post-training quantization (PTQ)** techniques.


### Run the Quantization + Compilation Config
Activate the **Quantization Python Environment** and run the workflow:

For Encoder Model:
```bash
olive run --config sam_vision_encoder_qnn.json
```

For Point and Box based Decoder Model:
```bash
olive run --config sam_mask_decoder_qnn_fp16_ctx.json
```

### Model ORT Execution

Execute SAM model in **AOT Compilation Python Environment** using following command:

```bash
python sam_mask_generator.py --model_ve path/to/encoder_model.onnx --model_md path/to/decoder_model.onnx --image_path car.png --box_x 40 --box_y 235 --box_w 940 --box_h 490 --output_path car_mask.png
```
