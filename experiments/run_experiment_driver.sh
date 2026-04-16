#!/bin/bash
# Experiment driver for standalone prompt optimization runs.
# Supports GEPA, GEPAFewShot, and MIPROv2 as standalone prompt optimizers.
#
# Usage:
#   bash experiments/run_experiment_driver.sh \
#       --dataset gsm8k \
#       --optimizer gepa_fewshot \
#       --model meta-llama/Llama-3.2-3B-Instruct

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f "$REPO_ROOT/../vm_vars.env" ]] && source "$REPO_ROOT/../vm_vars.env"
source "$REPO_ROOT/../dspy_venv/bin/activate"

# Defaults
DATASET="gsm8k"
OPTIMIZER="gepa_fewshot"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
AUTO="medium"
K_DEMOS=3
MAX_BOOTSTRAPPED_DEMOS=16
MAX_LABELED_DEMOS=4
DEMO_MUTATION_STRATEGY="metric_based"
TRAIN_SIZE=200
VAL_SIZE=100
TEST_SIZE=300
NUM_THREADS=4
API_BASE="http://localhost:30000/v1"
MAX_TOKENS=""         # empty = no limit (model decides)
REFLECTION_MODEL=""   # empty = same as task model
REFLECTION_MINIBATCH_SIZE=10
SEED=""           # empty = time-based (non-deterministic)
LOG_LEVEL="INFO"

# Supported values
VALID_DATASETS=("gsm8k" "iris")
VALID_OPTIMIZERS=("baseline" "gepa" "gepa_merge" "gepa_fewshot" "miprov2")
VALID_AUTO=("light" "medium" "heavy")
VALID_MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "microsoft/Phi-4-mini-instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3.5-0.8B"
    "Qwen/Qwen3.5-9B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "google/gemma-3-4b-it"
)

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset)                  DATASET="$2";                    shift ;;
        --optimizer)                OPTIMIZER="$2";                  shift ;;
        --model)                    MODEL="$2";                      shift ;;
        --auto)                     AUTO="$2";                       shift ;;
        --k-demos)                  K_DEMOS="$2";                    shift ;;
        --max-bootstrapped-demos)   MAX_BOOTSTRAPPED_DEMOS="$2";     shift ;;
        --max-labeled-demos)        MAX_LABELED_DEMOS="$2";          shift ;;
        --demo-mutation-strategy)   DEMO_MUTATION_STRATEGY="$2";     shift ;;
        --train-size)               TRAIN_SIZE="$2";                 shift ;;
        --val-size)                 VAL_SIZE="$2";                   shift ;;
        --test-size)                TEST_SIZE="$2";                  shift ;;
        --num-threads)              NUM_THREADS="$2";                shift ;;
        --api-base)                 API_BASE="$2";                   shift ;;
        --max-tokens)               MAX_TOKENS="$2";                 shift ;;
        --reflection-model)         REFLECTION_MODEL="$2";           shift ;;
        --reflection-minibatch-size) REFLECTION_MINIBATCH_SIZE="$2"; shift ;;
        --log-level)                LOG_LEVEL="$2";                  shift ;;
        --seed)                     SEED="$2";                       shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "  --dataset                  Dataset. Options: ${VALID_DATASETS[*]}"
            echo "  --optimizer                Optimizer. Options: ${VALID_OPTIMIZERS[*]}"
            echo "  --model                    Task LM identifier"
            echo "  --auto                     Budget preset. Options: ${VALID_AUTO[*]}. Default: medium"
            echo "  --k-demos                  Demos per candidate (GEPAFewShot). Default: 3"
            echo "  --max-bootstrapped-demos   Max bootstrapped demos in pool. Default: 16"
            echo "  --max-labeled-demos        Max labeled demos in pool. Default: 4"
            echo "  --demo-mutation-strategy   Demo mutation. Options: random, metric_based. Default: metric_based"
            echo "  --train-size               Training examples. Default: 200"
            echo "  --val-size                 Validation examples. Default: 100"
            echo "  --test-size                Test examples. Default: 300"
            echo "  --num-threads              Parallel eval threads. Default: 4"
            echo "  --api-base                 SGLang endpoint. Default: http://localhost:30000/v1"
            echo "  --max-tokens               Max tokens per LM call. Default: unset (model decides)"
            echo "  --reflection-model         GEPA reflection LM. Default: same as --model"
            echo "  --reflection-minibatch-size  Subsample size for GEPA reflection gate. Default: 25"
            echo "  --log-level                Logging verbosity: DEBUG/INFO/WARNING/ERROR. Default: INFO"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate
if [[ ! " ${VALID_DATASETS[*]} " =~ " ${DATASET} " ]]; then
    echo "Invalid dataset: $DATASET. Supported: ${VALID_DATASETS[*]}"; exit 1
fi
if [[ ! " ${VALID_OPTIMIZERS[*]} " =~ " ${OPTIMIZER} " ]]; then
    echo "Invalid optimizer: $OPTIMIZER. Supported: ${VALID_OPTIMIZERS[*]}"; exit 1
fi
if [[ ! " ${VALID_AUTO[*]} " =~ " ${AUTO} " ]]; then
    echo "Invalid --auto: $AUTO. Supported: ${VALID_AUTO[*]}"; exit 1
fi

# Auto-confirm MIPROv2 prompt (suppresses interactive y/n)
if [[ "$OPTIMIZER" == "miprov2" ]]; then
    export PYTHONUNBUFFERED=1
    export AUTO_CONFIRM=true
fi

# Build extra CLI flags
EXTRA_ARGS=""
if [[ -n "$REFLECTION_MODEL" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --reflection-model $REFLECTION_MODEL"
fi
if [[ -n "$MAX_TOKENS" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --max-tokens $MAX_TOKENS"
fi
if [[ -n "$SEED" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --seed $SEED"
fi
if [[ "$OPTIMIZER" == "gepa_fewshot" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --k-demos $K_DEMOS --demo-mutation-strategy $DEMO_MUTATION_STRATEGY"
fi

# Log file
LOG_FILE="$REPO_ROOT/experiments/logs/experiment__${DATASET}__${OPTIMIZER}__$(basename "$MODEL")__${AUTO}__$(date +'%Y-%m-%d').log"
mkdir -p "$REPO_ROOT/experiments/logs"

# Raise open-files limit (prevents LiteLLM SQLite exhaustion)
ulimit -n 65535

# Run
nohup python3.11 "$REPO_ROOT/experiments/run_experiment.py" \
    --dataset        "$DATASET" \
    --optimizer      "$OPTIMIZER" \
    --model          "$MODEL" \
    --auto           "$AUTO" \
    --train-size     "$TRAIN_SIZE" \
    --val-size       "$VAL_SIZE" \
    --test-size      "$TEST_SIZE" \
    --num-threads    "$NUM_THREADS" \
    --api-base       "$API_BASE" \
    --max-bootstrapped-demos    "$MAX_BOOTSTRAPPED_DEMOS" \
    --max-labeled-demos         "$MAX_LABELED_DEMOS" \
    --reflection-minibatch-size "$REFLECTION_MINIBATCH_SIZE" \
    --log-level                 "$LOG_LEVEL" \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_FILE" &

echo ""
echo "Experiment launched"
echo "-------------------"
echo "  Dataset    : $DATASET"
echo "  Optimizer  : $OPTIMIZER"
echo "  Model      : $MODEL"
echo "  Budget     : $AUTO"
echo "  Threads    : $NUM_THREADS"
echo "  Log file   : $LOG_FILE"
echo ""
