# Run multiple VLM recipes sequentially on a single runner.
# Each model is optimized for both CPU and GPU, evaluated, then cleaned up.
# Usage: .\run_sequential_vision_eval.ps1

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path "$ScriptDir\..\..").Path
$EvalScript = Join-Path $ScriptDir "run_vision_eval.py"

# Define models to run: recipe_dir, pytorch_model, limit
$Models = @(
    @{ Dir = "Qwen-Qwen3.5-0.8B"; PyTorch = "Qwen/Qwen3.5-0.8B"; Limit = 100 },
    @{ Dir = "Qwen-Qwen3.5-2B"; PyTorch = "Qwen/Qwen3.5-2B"; Limit = 100 },
    @{ Dir = "Qwen-Qwen3-VL-2B-Instruct"; PyTorch = "Qwen/Qwen3-VL-2B-Instruct"; Limit = 100 },
    @{ Dir = "Qwen-Qwen3-VL-4B-Instruct"; PyTorch = "Qwen/Qwen3-VL-4B-Instruct"; Limit = 100 },
    @{ Dir = "Qwen-Qwen3-VL-8B-Instruct"; PyTorch = "Qwen/Qwen3-VL-8B-Instruct"; Limit = 100 }
)

# Device configs: config_dir, optimize_device, eval_device
$Devices = @(
    @{ ConfigDir = "cpu_and_mobile"; OptimizeDevice = "cpu"; EvalDevice = "cpu" },
    @{ ConfigDir = "cuda"; OptimizeDevice = "gpu"; EvalDevice = "gpu" }
)

$Failed = @()
$Passed = @()

foreach ($model in $Models) {
    $recipeDir = $model.Dir
    $pytorchModel = $model.PyTorch
    $limit = $model.Limit

    $builtinDir = Join-Path $RepoRoot "$recipeDir\builtin"
    Set-Location $builtinDir

    # Install recipe-specific requirements if different from base
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt --quiet 2>&1 | Select-Object -Last 3
    }

    foreach ($dev in $Devices) {
        $configDir = $dev.ConfigDir
        $optimizeDevice = $dev.OptimizeDevice
        $evalDevice = $dev.EvalDevice
        $runLabel = "$recipeDir ($evalDevice)"

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "Starting: $runLabel"
        Write-Host "============================================================"

        # Optimize
        python optimize.py --config-dir $configDir --device $optimizeDevice
        if ($LASTEXITCODE -eq 0) {
            # Evaluate
            python $EvalScript --model-path "$configDir\models" --limit $limit --device $evalDevice --pytorch-model $pytorchModel
            if ($LASTEXITCODE -eq 0) {
                $Passed += $runLabel
            } else {
                Write-Host "WARN: Evaluation failed for $runLabel"
                $Failed += "$runLabel (eval)"
            }
        } else {
            Write-Host "WARN: Optimization failed for $runLabel"
            $Failed += "$runLabel (optimize)"
        }

        # Clean up model files to free memory
        Write-Host "Cleaning up $runLabel models..."
        $modelsPath = Join-Path $builtinDir "$configDir\models"
        if (Test-Path $modelsPath) {
            Remove-Item -Recurse -Force $modelsPath
        }
    }

    Set-Location $RepoRoot
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Sequential Run Summary"
Write-Host "============================================================"
Write-Host "  Passed: $($Passed.Count)/$($Models.Count)"
foreach ($p in $Passed) { Write-Host "    + $p" }
if ($Failed.Count -gt 0) {
    Write-Host "  Failed: $($Failed.Count)/$($Models.Count)"
    foreach ($f in $Failed) { Write-Host "    x $f" }
    exit 1
}
