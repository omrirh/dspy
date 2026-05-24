#!/bin/bash
#
# run_bt_matrix.sh — Run the BetterTogether prompt-optimizer matrix for the paper.
#
# Matrix: 1 model x 2 datasets x 3 optimizers x 1 seed + 2 baselines = 8 runs
#   Model:      Qwen/Qwen2.5-32B-Instruct-AWQ
#   Datasets:   hotpotqa, iris
#   Optimizers: clusterfs, miprov2, bfrs
#   Seeds:      100
#   Baselines:  1 per dataset (seed 100)
#   Strategy:   p (prompt-only, no finetuning)
#
# Usage:
#   ./run_bt_matrix.sh                           # run all 20 experiments
#   ./run_bt_matrix.sh --dry-run                 # preview commands only
#   ./run_bt_matrix.sh --resume                  # skip runs with existing JSON
#   ./run_bt_matrix.sh --sglang-port 7501        # override sglang port (for LM)
#   ./run_bt_matrix.sh --results-dir my_results  # override results directory
#   ./run_bt_matrix.sh --datasets hotpotqa       # run only specified dataset(s)
#   ./run_bt_matrix.sh --optimizers clusterfs bfrs  # run only specified optimizers

set -euo pipefail

source ../vm_vars.env
source ../dspy_venv/bin/activate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
DATASETS=("hotpotqa" "iris")
OPTIMIZERS=("clusterfs" "miprov2" "bfrs")
SEEDS=(100)
BASELINE_SEED=100
STRATEGY="p"

SGLANG_PORT="7501"
RESULTS_DIR="results_bt"

DRY_RUN=false
RESUME=false
DATASETS_OVERRIDE=()
OPTIMIZERS_OVERRIDE=()

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run)        DRY_RUN=true ;;
        --resume)         RESUME=true ;;
        --sglang-port)    SGLANG_PORT="$2"; shift ;;
        --results-dir)    RESULTS_DIR="$2"; shift ;;
        --datasets)       shift; while [[ "$#" -gt 0 && "$1" != --* ]]; do DATASETS_OVERRIDE+=("$1"); shift; done; continue ;;
        --optimizers)     shift; while [[ "$#" -gt 0 && "$1" != --* ]]; do OPTIMIZERS_OVERRIDE+=("$1"); shift; done; continue ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Preview commands without running"
            echo "  --resume          Skip runs where result JSON already exists"
            echo "  --sglang-port     sglang server port (default: 7501)"
            echo "  --results-dir     Output directory (default: results_bt)"
            echo "  --datasets        Space-separated dataset names (default: hotpotqa iris)"
            echo "                    e.g. --datasets hotpotqa"
            echo "  --optimizers      Space-separated optimizer names (default: clusterfs miprov2 bfrs)"
            echo "                    e.g. --optimizers clusterfs bfrs"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

[[ ${#DATASETS_OVERRIDE[@]} -gt 0 ]] && DATASETS=("${DATASETS_OVERRIDE[@]}")
[[ ${#OPTIMIZERS_OVERRIDE[@]} -gt 0 ]] && OPTIMIZERS=("${OPTIMIZERS_OVERRIDE[@]}")

MODEL_BASENAME=$(basename "$MODEL")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +'%Y-%m-%d_%H%M%S')
LOG_FILE="bt_matrix_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Required for MIPROv2 non-interactive runs
export PYTHONUNBUFFERED=1
export AUTO_CONFIRM=true

# Increase open file limit to prevent LiteLLM database errors
ulimit -n 65535

echo "============================================================"
echo "  BetterTogether Experiment Matrix"
echo "  Started: $(date)"
echo "  Log: $LOG_FILE"
echo "============================================================"
echo ""
echo "Model:      $MODEL"
echo "Datasets:   ${DATASETS[*]}"
echo "Optimizers: ${OPTIMIZERS[*]}"
echo "Seed:       ${SEEDS[*]}"
echo "Strategy:   $STRATEGY"
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
    local dataset="$1"
    local optimizer="$2"
    local seed="$3"
    local is_baseline="$4"

    # Determine expected JSON path
    local json_dir
    if [[ "$is_baseline" == "true" ]]; then
        json_dir="${RESULTS_DIR}/${dataset}/${MODEL_BASENAME}/baseline"
    else
        json_dir="${RESULTS_DIR}/${dataset}/${MODEL_BASENAME}/${optimizer}"
    fi
    local json_path="${json_dir}/${seed}.json"

    # Resume: skip if JSON already exists
    if [[ "$RESUME" == "true" && -f "$json_path" ]]; then
        echo "[SKIP] $json_path already exists"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi

    # Build command
    local cmd
    if [[ "$is_baseline" == "true" ]]; then
        cmd=(
            python3.11 better_together_experiment.py
            --dataset "$dataset"
            --prompt-optimizer "clusterfs"
            --strategy "$STRATEGY"
            --model "$MODEL"
            --seed "$seed"
            --results-dir "$RESULTS_DIR"
            --baseline
        )
    else
        cmd=(
            python3.11 better_together_experiment.py
            --dataset "$dataset"
            --prompt-optimizer "$optimizer"
            --strategy "$STRATEGY"
            --model "$MODEL"
            --seed "$seed"
            --results-dir "$RESULTS_DIR"
        )
    fi

    RUN_COUNT=$((RUN_COUNT + 1))

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN #${RUN_COUNT}] ${cmd[*]}"
        return 0
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[RUN #${RUN_COUNT}] dataset=${dataset} optimizer=${optimizer} seed=${seed} baseline=${is_baseline}"
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

        if [[ -f "$json_path" ]]; then
            echo "[OK] JSON written: $json_path"
            if command -v jq &>/dev/null; then
                local version
                version=$(jq -r '.schema_version // empty' "$json_path" 2>/dev/null)
                [[ "$version" == "2.0" ]] && echo "[OK] Schema version: $version" || echo "[WARN] Unexpected schema: ${version:-missing}"
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
# Run matrix — grouped by dataset to keep context consistent
# ---------------------------------------------------------------------------
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Dataset block: $dataset"
    echo "============================================================"

    # Baseline first (must exist before optimizer runs for delta computation)
    run_experiment "$dataset" "baseline" "$BASELINE_SEED" "true"

    # Optimizer × seed combinations
    for optimizer in "${OPTIMIZERS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "$dataset" "$optimizer" "$seed" "false"
        done
    done
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
