## Stable Diffusion Optimization with ONNX Runtime VitisAI EP

This folder contains sample Olive configurations to optimize **Stable Diffusion v1.5** subgraphs for the **VitisAI Execution Provider** on AMD NPU.

## Supported models and configs

| Model ID (Hugging Face) | Config file |
|:---------------------|:------------|
| `sd-legacy/stable-diffusion-v1-5` | `config_unet.json` |
| `sd-legacy/stable-diffusion-v1-5` | `config_vae_decoder.json` |
| `sd-legacy/stable-diffusion-v1-5` | `config_vae_encoder.json` |
| `sd-legacy/stable-diffusion-v1-5` | `config_text_encoder.json` |
| `sd-legacy/stable-diffusion-v1-5` | `config_safety_checker.json` |

## Run the VitisAI workflow

#### Create a conda environment and install Olive

```bash
conda create -n olive python=3.12
conda activate olive
```

```bash
cd Olive
pip install -e .
pip install -r requirements.txt
```

#### Install VitisAI Stable Diffusion dependencies

```bash
cd olive-recipes/sd-legacy-stable-diffusion-v1-5/VitisAI
pip install --force-reinstall -r requirements_vitisai_sd.txt
```

#### Generate optimized subgraphs

Run Olive from the **`VitisAI`** recipe directory so `user_script.py` and model assets resolve correctly:

```bash
cd olive-recipes/sd-legacy-stable-diffusion-v1-5/VitisAI

olive run --config ../VitisAI/config_unet.json
olive run --config ../VitisAI/config_vae_decoder.json
olive run --config ../VitisAI/config_vae_encoder.json
olive run --config ../VitisAI/config_text_encoder.json
olive run --config ../VitisAI/config_safety_checker.json
```

Optimized artifacts are written to the `output_dir` defined in each JSON (for example `footprints/unet`, `footprints/vae_decoder`, …).

> **Note:** Exact paths depend on `output_dir` and `cache_dir` in each config file.

### Execution provider and hardware placement

| Component | Execution provider | Compute device |
|-----------|-------------------|----------------|
| UNet | VitisAI EP | NPU |
| VAE decoder | VitisAI EP | NPU |
| Text encoder | CPU EP | CPU |
| VAE encoder | CPU EP | CPU |
| Safety checker | CPU EP | CPU |

The VitisAI Execution Provider is used only for the **UNet** and **VAE decoder**. All other subgraphs run with the **CPU Execution Provider** on the host CPU.

### End-to-end image generation (inference)

```bash
cd olive-recipes/sd-legacy-stable-diffusion-v1-5/VitisAI

python stable_diffusion.py --provider vitisai --model_id sd-legacy/stable-diffusion-v1-5 --seed 0 --guidance_scale 7.5 --num_inference_steps 20 --prompt "Photo of an ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner."
```

## Outputs (relative to `olive/`)

| Item | Location |
|:-----|:---------|
| Generated images | `result_0.png`, `result_1.png`, … in the **current working directory** (typically `VitisAI/` if you run the command from there) |
| Full pipeline, unoptimized | `model/unoptimized/<model_id>/` |
| Full pipeline, optimized (VitisAI) | `model/optimized-vitisai/<model_id>/` |

`model_id` slashes become nested folders (e.g. `sd-legacy/stable-diffusion-v1-5`). Per-subgraph `olive run` outputs use each config’s `output_dir` / `cache_dir` (e.g. under `footprints/`, `vai_cache/`).
