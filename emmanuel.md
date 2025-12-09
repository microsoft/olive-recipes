# CI Workflow Consolidation - Changes Summary

## Overview
Consolidated the CI automation from two separate workflows (`ci.yml` and `main.yml`) into a single, scalable solution using `olive_ci.json` configuration files.

## What Was Done

### 1. Removed Old Workflow
**Deleted:** `.github/workflows/ci.yml`
- This workflow had hardcoded test matrices
- Required manual updates when adding new recipes
- Had 217 lines of configuration

### 2. Updated Main Workflow
**Modified:** `.github/workflows/main.yml`
- Added weekly cron schedule: `'0 15 * * 5'` (Fridays at 3 PM UTC)
- Now runs automatically every week, plus on-demand when `olive_ci.json` files change
- Dynamically discovers all recipes via `generate_matrix.py` script

### 3. Created/Updated olive_ci.json Files

#### `intel-bert-base-uncased-mrpc/oci/olive_ci.json`
**Updated** to include all tests from `ci.yml`:
- PTQ (Ubuntu/CPU)
- INC Smooth Quant (Ubuntu/CPU) ← **Added**
- PTQ (Ubuntu/CUDA)
- PTQ (Windows/CPU)

#### `microsoft-resnet-50/olive/olive_ci.json`
**Created** with:
- ResNet CPU Session Param Tuning (Ubuntu/CPU)

#### `meta-llama-Llama-3.2-1B-Instruct/olive/olive_ci.json`
**Created** with all 8 GPU tests (previously commented out in `ci.yml`):
- Dora (Ubuntu/CUDA)
- HQQ (Ubuntu/CUDA)
- Lmeval (Ubuntu/CUDA)
- Lmeval-ONNX (Ubuntu/CUDA)
- Loha (Ubuntu/CUDA)
- Lokr (Ubuntu/CUDA)
- QLoRA (Ubuntu/CUDA)
- RTN (Ubuntu/CUDA)

## How It Works Now

### Automatic Discovery
The `main.yml` workflow:
1. Scans the entire repository for `olive_ci.json` files
2. Generates test matrices dynamically for each OS/device combination
3. Creates GitHub Actions jobs automatically

### Adding New Recipes
To add a new recipe to CI, simply:
1. Create an `olive_ci.json` file in the model's directory
2. Define the test configuration (name, os, device, requirements_file, command)
3. Commit the file - CI will automatically pick it up

### Example olive_ci.json Format
```json
[
    {
        "name": "test_name",
        "os": "ubuntu",
        "device": "cpu",
        "requirements_file": "requirements.txt",
        "command": "python -m olive run --config config.json"
    }
]
```

## Benefits

1. **Single Source of Truth**: One workflow handles all CI needs
2. **Scalability**: Add new recipes without touching workflow files
3. **Weekly Automation**: Runs every Friday to catch regressions
4. **Flexibility**: Each model owns its CI configuration
5. **Less Maintenance**: No need to update workflow YAML for new tests

## Caching
The CI already has optimal caching configured:
- **pip packages** cached based on requirements.txt hash
- **HuggingFace models** cached to avoid re-downloads
- Works seamlessly with 1ES self-hosted runners
- 10 GB cache limit per repository, 7-day retention

## Branch
All changes committed to: `ci-test` branch
