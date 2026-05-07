#!/bin/bash

python stable_diffusion.py --model_id sd-legacy/stable-diffusion-v1-5 --provider cpu --format qdq --optimize --only_conversion
python evaluation.py --save_data --model_id sd-legacy/stable-diffusion-v1-5 --num_inference_steps 25 --seed 0 --num_data 100 --guidance_scale 7.5
python stable_diffusion.py --model_id sd-legacy/stable-diffusion-v1-5 --provider qnn --format qdq --optimize
