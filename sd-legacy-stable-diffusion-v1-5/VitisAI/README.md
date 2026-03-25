## Stable Diffusion Optimization with ONNX Runtime VitisAI EP

### Prerequisites
```bash
python -m pip install -r requirements_vitisai_sd.txt
```

### Run Pipeline to generate one image:

`python .\stable_diffusion.py --provider vitisai --model_id sd-legacy/stable-diffusion-v1-5 --guidance_scale 7.5 --num_inference_steps 20 --prompt "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner."`

