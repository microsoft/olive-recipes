## Stable Diffusion Optimization with ONNX Runtime QNN EP

### Prerequisites
```bash
python -m pip install -r requirements_qnn.txt
```
### Generate data for static quantization

To get better result, we need to generate real data from original model instead of using random data for static quantization.

First generate onnx unoptimized model:

`python stable_diffusion.py --model_id sd-legacy/stable-diffusion-v1-5 --provider cpu --format qdq --optimize --only_conversion`

Then generate data:

`python .\evaluation.py --save_data --model_id sd-legacy/stable-diffusion-v1-5 --num_inference_steps 25 --seed 0 --num_data 100 --guidance_scale 7.5`

### Optimize

Optimize the onnx models for performance improvements. vae_decoder and unet are per-channel quantized and text_encoder runs in fp16 precision.

`python stable_diffusion.py --model_id sd-legacy/stable-diffusion-v1-5 --provider qnn --format qdq --optimize`

### Test and evaluate

`python .\evaluation.py --model_id sd-legacy/stable-diffusion-v1-5 --num_inference_steps 25 --seed 0 --num_data 100 --guidance_scale 7.5 --provider QNNExecutionProvider --model_dir optimized-qnn_qdq`

To generate one image:

`python stable_diffusion.py --model_id sd-legacy/stable-diffusion-v1-5 --provider qnn --format qdq --guidance_scale 7.5 --seed 0 --num_inference_steps 25 --prompt "A baby is laying down with a teddy bear"`

### References

[stable-diffusion-v1-4](https://github.com/microsoft/olive-recipes/tree/main/compvis-stable-diffusion-v1-4/olive#readme)
