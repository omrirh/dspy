#!/bin/bash
#
# run_react_matrix.sh — Run the full ReAct experiment matrix.
#
# Matrix: 3 models x 3 optimizers x 3 seeds + 3 baselines = 30 runs
#   Models:     Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct,
#               meta-llama/Llama-3.1-8B-Instruct
#   Optimizers: clusterfs, miprov2, bfrs
#   Seeds:      100, 200, 300
#   Baselines:  1 per model (seed 100)
#
# Usage:
#   ./run_react_matrix.sh                          # run all 30 experiments
#   ./run_react_matrix.sh --dry-run                # preview commands only
#   ./run_react_matrix.sh --resume                 # skip runs with existing JSON
#   ./run_react_matrix.sh --sglang-port 7501       # override sglang port
#   ./run_react_matrix.sh --results-dir my_results # override results directory
#   ./run_react_matrix.sh --models "Qwen/Qwen2.5-7B-Instruct"  # run single model

set -euo pipefail

source ../vm_vars.env
source ../dspy_venv/bin/activate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)
OPTIMIZERS=("clusterfs" "miprov2" "bfrs")
SEEDS=(100 200 300)
BASELINE_SEED=100

COLBERT_URL="http://localhost:8894/api/search"
SGLANG_PORT="7501"
ENCODER_DEVICE="cpu"
TRAIN_SIZE=100
DEV_SIZE=250
TEST_SIZE=1500
MAX_ITERS=20
RESULTS_DIR="results"

DRY_RUN=false
RESUME=false
MODELS_OVERRIDE=()

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run)        DRY_RUN=true ;;
        --resume)         RESUME=true ;;
        --sglang-port)    SGLANG_PORT="$2"; shift ;;
        --colbert-url)    COLBERT_URL="$2"; shift ;;
        --encoder-device) ENCODER_DEVICE="$2"; shift ;;
        --results-dir)    RESULTS_DIR="$2"; shift ;;
        --train-size)     TRAIN_SIZE="$2"; shift ;;
        --dev-size)       DEV_SIZE="$2"; shift ;;
        --test-size)      TEST_SIZE="$2"; shift ;;
        --max-iters)      MAX_ITERS="$2"; shift ;;
        --models)         IFS=',' read -ra MODELS_OVERRIDE <<< "$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Preview commands without running"
            echo "  --resume          Skip runs where result JSON already exists"
            echo "  --sglang-port     sglang server port (default: 7501)"
            echo "  --colbert-url     ColBERTv2 endpoint (default: $COLBERT_URL)"
            echo "  --encoder-device  SentenceTransformer device (default: cpu)"
            echo "  --results-dir     Output directory (default: results)"
            echo "  --train-size      Training examples (default: 100)"
            echo "  --dev-size        Validation examples (default: 250)"
            echo "  --test-size       Test examples (default: 1500)"
            echo "  --max-iters       Max ReAct steps (default: 20)"
            echo "  --models          Comma-separated model IDs to run (default: all 3)"
            echo "                    e.g. --models 'meta-llama/Llama-3.1-8B-Instruct'"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Apply model override if provided
[[ ${#MODELS_OVERRIDE[@]} -gt 0 ]] && MODELS=("${MODELS_OVERRIDE[@]}")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +'%Y-%m-%d_%H%M%S')
LOG_FILE="react_matrix_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Required for MIPROv2 non-interactive runs
export PYTHONUNBUFFERED=1
export AUTO_CONFIRM=true

echo "============================================================"
echo "  ReAct Experiment Matrix"
echo "  Started: $(date)"
echo "  Log: $LOG_FILE"
echo "============================================================"
echo ""
echo "Models:     ${MODELS[*]}"
echo "Optimizers: ${OPTIMIZERS[*]}"
echo "Seeds:      ${SEEDS[*]}"
echo "Dry run:    $DRY_RUN"
echo "Resume:     $RESUME"
echo "Results:    $RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RUN_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

run_experiment() {
    local model="$1"
    local optimizer="$2"
    local seed="$3"
    local is_baseline="$4"
    local model_basename
    model_basename=$(basename "$model")

    # Determine expected JSON path
    local json_dir
    if [[ "$is_baseline" == "true" ]]; then
        json_dir="${RESULTS_DIR}/${model_basename}/baseline"
    else
        json_dir="${RESULTS_DIR}/${model_basename}/${optimizer}"
    fi
    local json_path="${json_dir}/${seed}.json"

    # Resume: skip if JSON already exists
    if [[ "$RESUME" == "true" && -f "$json_path" ]]; then
        echo "[SKIP] $json_path already exists"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi

    # Build command
    local cmd=(
        python3.11 react_agent_experiment.py
        --model "$model"
        --optimizer "$optimizer"
        --colbert-url "$COLBERT_URL"
        --sglang-port "$SGLANG_PORT"
        --train-size "$TRAIN_SIZE"
        --dev-size "$DEV_SIZE"
        --test-size "$TEST_SIZE"
        --max-iters "$MAX_ITERS"
        --encoder-device "$ENCODER_DEVICE"
        --seed "$seed"
        --results-dir "$RESULTS_DIR"
        --no-visuals
    )
    [[ "$is_baseline" == "true" ]] && cmd+=(--baseline)

    RUN_COUNT=$((RUN_COUNT + 1))

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN #${RUN_COUNT}] ${cmd[*]}"
        return 0
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[RUN #${RUN_COUNT}] model=${model_basename} optimizer=${optimizer} seed=${seed} baseline=${is_baseline}"
    echo "  Command: ${cmd[*]}"
    echo "  Started: $(date)"
    echo "------------------------------------------------------------"

    local run_start
    run_start=$(date +%s)

    if "${cmd[@]}"; then
        local run_end
        run_end=$(date +%s)
        local elapsed=$((run_end - run_start))
        echo "[DONE #${RUN_COUNT}] Completed in ${elapsed}s"

        # Validate JSON exists
        if [[ -f "$json_path" ]]; then
            echo "[OK] JSON written: $json_path"
            # Basic schema check with jq if available
            if command -v jq &>/dev/null; then
                local version
                version=$(jq -r '.schema_version // empty' "$json_path" 2>/dev/null)
                if [[ "$version" == "1.0" ]]; then
                    echo "[OK] Schema version: $version"
                else
                    echo "[WARN] Unexpected schema version: ${version:-missing}"
                fi
            fi
        else
            echo "[WARN] Expected JSON not found: $json_path"
        fi
    else
        echo "[FAIL #${RUN_COUNT}] Exit code $?"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ---------------------------------------------------------------------------
# Run matrix — grouped by model to minimize SGLang server swaps
# ---------------------------------------------------------------------------
for model_idx in "${!MODELS[@]}"; do
    model="${MODELS[$model_idx]}"
    model_basename=$(basename "$model")

    echo ""
    echo "============================================================"
    echo "  Model block: $model_basename"
    echo "============================================================"

    # Baseline (once per model)
    run_experiment "$model" "clusterfs" "$BASELINE_SEED" "true"

    # Optimizer × seed combinations
    for optimizer in "${OPTIMIZERS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "$model" "$optimizer" "$seed" "false"
        done
    done

    # Pause between model blocks for server swap (except after last)
    if [[ "$DRY_RUN" != "true" && $model_idx -lt $((${#MODELS[@]} - 1)) ]]; then
        echo ""
        echo "============================================================"
        echo "  Model block complete: $model_basename"
        echo "  Next model requires server swap."
        echo "  Press ENTER to continue or Ctrl-C to abort..."
        echo "============================================================"
        read -r
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Matrix complete"
echo "  Finished: $(date)"
echo "  Runs: $RUN_COUNT  Skipped: $SKIP_COUNT  Failed: $FAIL_COUNT"
echo "  Results: $RESULTS_DIR/"
echo "  Log: $LOG_FILE"
echo "============================================================"

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
