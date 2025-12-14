#!/bin/bash

# Enable session variables and env
source ../vm_vars.env
source ../dspy_venv/bin/activate

# Default values
DATASET="gsm8k"
PROMPT_OPTIMIZER="clusterfs"
STRATEGY="p"
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# Supported values
VALID_DATASETS=("hotpotqa" "gsm8k" "iris")
VALID_PROMPT_OPTIMIZERS=("bfrs" "clusterfs" "miprov2" "gepa")
VALID_STRATEGIES=("p" "w" "p -> w" "w -> p" "p -> w -> p" "p -> p" "p -> p -> p")
VALID_MODELS=(
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "Qwen/Qwen2.5-7B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "gemini/gemini-2.5-flash"
  "gemini/gemini-2.5-pro"
  "Qwen/Qwen3-8B"
  "google/gemma-3-4b-it"
  "Qwen/Qwen2-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "openai/gpt-oss-20b"
)

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --prompt-optimizer) PROMPT_OPTIMIZER="$2"; shift ;;
        --strategy) STRATEGY="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--dataset <dataset name>] [--prompt-optimizer <optimizer>] [--strategy <strategy>] [--model <model name>]"
            echo "  --dataset           Specify the dataset to use. Options: hotpotqa, gsm8k, iris"
            echo "  --prompt-optimizer  Specify the prompt optimization method. Default: bfrs. Options: bfrs, clusterfs, miprov2, gepa"
            echo "  --strategy          Specify the strategy. Default: p. Options: 'p', 'w', 'p -> p', 'p -> p -> p', 'p -> w', 'w -> p', 'p -> w -> p'"
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

# Validate dataset
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Invalid model: $MODEL"
    echo "Supported models: ${VALID_MODELS[*]}"
    exit 1
fi

# Auto confirm prompt optimization run with MIPROv2
if [[ "$PROMPT_OPTIMIZER" == "miprov2" ]]; then
    export PYTHONUNBUFFERED=1
    export AUTO_CONFIRM=true
fi

# Format additional params for log filename
LOG_PARAMS="_${PROMPT_OPTIMIZER}_$(echo "$STRATEGY" | tr ' ' '_')"
MODEL_ID="_$(basename "$MODEL")"

# Set log file
EXPERIMENT_LOG_FILE="better_together_experiment_run_${DATASET}${MODEL_ID}${LOG_PARAMS}_$(date +'%Y-%m-%d').log"

# PATCH: set higher number of open files limit to prevent LiteLLM database limit reach
ulimit -n 65535

# Run experiment
nohup python3.11 better_together_experiment.py \
    --dataset "$DATASET" \
    --prompt-optimizer "$PROMPT_OPTIMIZER" \
    --strategy "$STRATEGY" \
    --model "$MODEL" 2>&1 | tee "$EXPERIMENT_LOG_FILE" &

echo -e "\nBetterTogether Experiment"
echo "-------------------------"
echo "Dataset: $DATASET"
echo "Prompt Optimizer: $PROMPT_OPTIMIZER"
echo "Strategy: $STRATEGY"
echo "Model: $MODEL"
echo -e "Log file: $EXPERIMENT_LOG_FILE\n\n"
