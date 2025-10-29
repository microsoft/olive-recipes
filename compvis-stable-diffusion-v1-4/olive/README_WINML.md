# Generate plain cpu version

`python -m pip install -r requirements-common.txt`

`python stable_diffusion.py --model_id stabilityai/stable-diffusion-2-1 --provider cpu --format qdq --optimize --only_conversion`

# Patch code

See https://github.com/huggingface/diffusers/pull/12555/files

# Generate a reference image

`uv run -p C:\Users\[User]\.aitk\bin\model_lab_runtime\Python-WCR-win32-x64-3.12.9 stable_diffusion.py --model_id stabilityai/stable-diffusion-2-1 --provider cpu --format qdq --seed 0 --num_inference_steps 50 --prompt "A baby is laying down with a teddy bear" --test_unoptimized`

# Test your model

Update onnx files in `olive-recipes\compvis-stable-diffusion-v1-4\olive\models\unoptimized\stabilityai\stable-diffusion-2-1` and `get_qdq_pipeline` in `olive-recipes\compvis-stable-diffusion-v1-4\olive\sd_utils\qdq.py`.
