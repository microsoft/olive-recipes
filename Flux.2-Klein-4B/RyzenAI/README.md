# FLUX.2-klein-4B — Model Optimization for AMD NPU

This folder contains sample Olive configurations and export script to optimize FLUX.2-klein-4B 2 models for AMD NPU.

The export script (`export_models.py`) handles the full pipeline in
one command: download weights → ONNX conversion → NPU compilation →
assemble a self-contained output directory (ONNX models + tokenizer + scheduler).

## Prerequisites

| Requirement | Notes |
|---|---|
| AMD NPU hardware | Ryzen AI device (NPU required for transformer / VAE decoder) |
| Windows 10/11 (x64) | Tested environment |
| Conda | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda |
| ~40 GB free disk | Weights + ONNX artifacts + Olive cache |
| HuggingFace account | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is a gated model — license acceptance required |

## Step 1 — Create the conda environment

```bash
conda create -n olive python=3.12 -y
conda activate olive
```

## Step 2 — Install model dependencies

```bash
git clone https://github.com/microsoft/olive-recipes.git
cd olive-recipes/Flux.2-Klein-4B/RyzenAI
pip install -r requirements_ryzenai_sd.txt
```

## Step 3 — Run the export

```bash
python export_models.py
```

The script will:

1. Download and cache the full pipeline weights from HuggingFace (first run only).
2. Export each sub-model to ONNX via Olive.
3. Compile the transformer and VAE decoder for AMD NPU using `VitisGenerateModelSD`.
4. Assemble the final pipeline directory, including non-ONNX components
   (tokenizer, scheduler).

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--model_id` | value stored in `config_*.json` | HuggingFace model ID or local path. Written back to all config files. |
| `--models` | all | Sub-models to export. Choices: `transformer vae_decoder text_encoder` |
| `--resolutions` | `1024x1024` | NPU compilation resolution(s). Written back to all config files. |
| `--output_dir` | `./output_model` | Destination for the assembled pipeline directory. |

```bash
# Export only the transformer
python export_models.py --models transformer

# Use a local model directory
python export_models.py --model_id D:/models/FLUX.2-klein-4B

# Change output directory
python export_models.py --output_dir D:/output/flux2_klein
```

## Output layout

After a successful export, `output_model/` (or your `--output_dir`) will contain:

```
output_model/
├── transformer/
│   ├── dd/
│   │   └── replaced.onnx      ← NPU-compiled
│   └── cache/
├── vae_decoder/
│   ├── dd/
│   │   └── replaced.onnx      ← NPU-compiled
│   └── cache/
├── text_encoder/
│   └── model.onnx             ← CPU ONNX
├── tokenizer/
└── scheduler/
```

> Olive writes intermediate outputs under `footprints/` and caches converted
> models in `cache/`. These can be safely deleted after the final pipeline is
> assembled.