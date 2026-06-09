#!/bin/bash
# Run multiple VLM recipes sequentially on a single runner.
# Each model is optimized, evaluated, then cleaned up before the next starts.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_vision_eval.py"

# Determine device from argument (default: cpu)
DEVICE="${1:-cpu}"
if [ "$DEVICE" = "gpu" ]; then
    CONFIG_DIR="cuda"
    OPTIMIZE_DEVICE="gpu"
else
    CONFIG_DIR="cpu_and_mobile"
    OPTIMIZE_DEVICE="cpu"
fi

# Define models to run: recipe_dir|pytorch_model|limit
MODELS=(
    "Qwen-Qwen3.5-0.8B|Qwen/Qwen3.5-0.8B|100"
    "Qwen-Qwen3.5-2B|Qwen/Qwen3.5-2B|100"
    "Qwen-Qwen3-VL-2B-Instruct|Qwen/Qwen3-VL-2B-Instruct|100"
    "Qwen-Qwen3-VL-4B-Instruct|Qwen/Qwen3-VL-4B-Instruct|100"
    "Qwen-Qwen3-VL-8B-Instruct|Qwen/Qwen3-VL-8B-Instruct|100"
)

FAILED=()
PASSED=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r recipe_dir pytorch_model limit <<< "$entry"

    echo ""
    echo "============================================================"
    echo "Starting: $recipe_dir"
    echo "============================================================"

    cd "$REPO_ROOT/$recipe_dir/builtin"

    # Install recipe-specific requirements if different from base
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt --quiet 2>&1 | tail -3
    fi

    # Optimize
    if python optimize.py --config-dir "$CONFIG_DIR" --device "$OPTIMIZE_DEVICE"; then
        # Evaluate
        if python "$EVAL_SCRIPT" --model-path "$CONFIG_DIR/models" --limit "$limit" --device "$DEVICE" --pytorch-model "$pytorch_model"; then
            PASSED+=("$recipe_dir")
        else
            echo "WARN: Evaluation failed for $recipe_dir"
            FAILED+=("$recipe_dir (eval)")
        fi
    else
        echo "WARN: Optimization failed for $recipe_dir"
        FAILED+=("$recipe_dir (optimize)")
    fi

    # Clean up model files to free memory for next model
    echo "Cleaning up $recipe_dir models..."
    rm -rf "$CONFIG_DIR/models"
    cd "$REPO_ROOT"
done

echo ""
echo "============================================================"
echo "Sequential Run Summary"
echo "============================================================"
echo "  Passed: ${#PASSED[@]}/${#MODELS[@]}"
for p in "${PASSED[@]}"; do echo "    ✓ $p"; done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${#FAILED[@]}/${#MODELS[@]}"
    for f in "${FAILED[@]}"; do echo "    ✗ $f"; done
    exit 1
fi
