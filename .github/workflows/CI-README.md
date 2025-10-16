# Olive Recipes CI

Automated CI pipeline for testing Olive recipe optimization workflows with the latest PyTorch.

## What It Does

- **Schedule**: Runs every Friday at 3:00 PM UTC (7 AM PT)
- **On Push**: Runs on every push to `recipes-ci` branch
- **Manual**: Trigger via GitHub Actions with scope options
- **Tests**: 5 CPU recipes across Linux and Windows (GPU tests temporarily disabled)
- **PyTorch**: Uses latest PyTorch automatically from requirements.txt
- **Artifacts**: Uploads optimized models with 7-day retention

## Recipes Tested

### **Linux CPU (3 tests)** 
- **PTQ** - Post-Training Quantization on Intel BERT (~1.6x speedup)
- **INC Smooth Quant** - Intel Neural Compressor Smooth Quantization (~2.3x speedup)
- **ResNet-50 Session Param Tuning** - Image classification optimization (~2.2x speedup, 87.89% accuracy)

### **Windows CPU (2 tests)** 
- **PTQ** - Post-Training Quantization on Intel BERT
- **INC Smooth Quant** - Known to fail due to upstream Olive/INC dtype mismatch bug

### **Linux GPU (8 tests)** Temporarily Disabled
- Dora, HQQ, Lmeval, Lmeval-ONNX, Loha, Lokr, QLoRA, RTN
- **Status**: Commented out, waiting for pool hardware upgrade
- **Issue**: Pool has Tesla M60 (CUDA 5.2) instead of A10 (CUDA 8.6)
- **Contact**: Xiaoyu/Jambay/Changming for pool SKU update to Standard_NC40ads_H100_v5
- **Ready**: Full configuration preserved, easy to re-enable once pool is fixed

## Test Flow

Each test performs:
1. Checkout code
2. Setup Python 3.10
3. Install Olive from main branch
4. Install optimum for HuggingFace exports
5. Install recipe requirements (includes latest PyTorch)
6. Print PyTorch version (for tracking)
7. Run Olive workflow with config
8. Upload artifacts (optimized models)

## Manual Testing

```bash
# Test recipe locally
cd <recipe_path>
python -m olive run --config <config>.json

# Verify output model
ls models/

# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Manual Trigger

```bash
# Run all tests
gh workflow run ci.yml

# Run specific scope
gh workflow run ci.yml -f test_scope=linux-cpu-only
gh workflow run ci.yml -f test_scope=windows-cpu-only
gh workflow run ci.yml -f test_scope=all
```

## Infrastructure

- **Linux CPU**: `ubuntu-latest` (GitHub-hosted)
- **Windows CPU**: `windows-latest` (GitHub-hosted)
- **Linux GPU**: `["self-hosted", "1ES.Pool=olive-github-ubuntu2204-cuda-A10"]` (disabled)
- **Timeout**: 120 minutes per job
- **Python**: 3.10
- **PyTorch**: Latest from PyPI (currently 2.9.0+cpu)

## Requirements

### Secrets
- `HUGGING_FACE_HUB_TOKEN` - Required for accessing gated HuggingFace models

### GPU Pool (When Re-enabled)
- Contact infrastructure team for pool upgrades
- Required: CUDA capability ≥ 7.0 (Volta or newer)
- Current issue: Tesla M60 has CUDA 5.2 (too old for PyTorch)

## Adding New Recipes

Edit `.github/workflows/ci.yml` and add to the appropriate job's matrix:

```yaml
matrix:
  test:
    - name: "My Recipe Name"
      path: "model-name/folder"
      config: "config.json"
```


