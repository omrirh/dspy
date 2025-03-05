#!/bin/bash

# Enable session variables and env
source ../vm_vars.env
source ../dspy_venv/bin/activate

# Default values
DATASET="gsm8k"
PROMPT_OPTIMIZER="bfrs"
STRATEGY="p"
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# Supported values
VALID_DATASETS=("hotpotqa" "gsm8k")
VALID_PROMPT_OPTIMIZERS=("bfrs" "clusterfs")
VALID_STRATEGIES=("p" "w" "p -> w" "w -> p" "p -> w -> p")

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --prompt-optimizer) PROMPT_OPTIMIZER="$2"; shift ;;
        --strategy) STRATEGY="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--dataset <dataset name>] [--prompt-optimizer <optimizer>] [--strategy <strategy>] [--model <model name>]"
            echo "  --dataset           Specify the dataset to use. Options: hotpotqa, gsm8k"
            echo "  --prompt-optimizer  Specify the prompt optimization method. Default: bfrs. Options: bfrs, clusterfs"
            echo "  --strategy          Specify the strategy. Default: p. Options: 'p', 'w', 'p -> w', 'w -> p', 'p -> w -> p'"
            echo "  --model             Specify the model to use. Default: meta-llama/Meta-Llama-3-8B-Instruct"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate dataset
if [[ ! " ${VALID_DATASETS[@]} " =~ " ${DATASET} " ]]; then
    echo "Invalid dataset: $DATASET"
    echo "Supported datasets: ${VALID_DATASETS[*]}"
    exit 1
fi

# Validate prompt-optimizer if provided
if [[ ! " ${VALID_PROMPT_OPTIMIZERS[@]} " =~ " ${PROMPT_OPTIMIZER} " ]]; then
    echo "Invalid prompt-optimizer: $PROMPT_OPTIMIZER"
    echo "Supported prompt-optimizers: ${VALID_PROMPT_OPTIMIZERS[*]}"
    exit 1
fi

# Validate strategy if provided
if [[ ! " ${VALID_STRATEGIES[@]} " =~ " ${STRATEGY} " ]]; then
    echo "Invalid strategy: $STRATEGY"
    echo "Supported strategies: ${VALID_STRATEGIES[*]}"
    exit 1
fi

# Format additional params for log filename
LOG_PARAMS="_${PROMPT_OPTIMIZER}_$(echo "$STRATEGY" | tr ' ' '_')"

# Set log file
EXPERIMENT_LOG_FILE="better_together_experiment_run_${DATASET}${LOG_PARAMS}_$(date +'%Y-%m-%d').log"

# Run experiment
nohup python3.11 better_together_experiment.py \
    --dataset "$DATASET" \
    --prompt-optimizer "$PROMPT_OPTIMIZER" \
    --strategy "$STRATEGY" \
    --model "$MODEL" 2>&1 | tee "$EXPERIMENT_LOG_FILE" &

echo "Experiment for dataset: $DATASET"
echo "Prompt Optimizer: $PROMPT_OPTIMIZER"
echo "Strategy: $STRATEGY"
echo "Model: $MODEL"
echo "Log file: $EXPERIMENT_LOG_FILE"
