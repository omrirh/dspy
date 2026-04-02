#!/bin/bash

# Enable session variables and env
source ../vm_vars.env
source ../dspy_venv/bin/activate

# Default values
MODEL="Qwen/Qwen2.5-7B-Instruct"
OPTIMIZER="clusterfs"
COLBERT_URL="http://localhost:8894/api/search"
SGLANG_PORT=""
TRAIN_SIZE=100
DEV_SIZE=250
TEST_SIZE=1500
MAX_ITERS=20
ENCODER_DEVICE="cpu"
BASELINE=false
NO_VISUALS=false
SAMPLE_TRAJECTORY=false
SEED=""

# Supported models
VALID_MODELS=(
  # 2B–8B — baseline / comparison
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen3-8B"
  "google/gemma-3-4b-it"
  # 14B–32B — agentic-ready, BF16 on A100 80GB
  "microsoft/Phi-4"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen3-32B"
  # 70B — FP8 on A100 80GB (sglang --quantization fp8 required)
  "meta-llama/Llama-3.3-70B-Instruct"
  # API models
  "gemini/gemini-2.5-flash"
  "gemini/gemini-2.5-pro"
)

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2"; shift ;;
        --optimizer)        OPTIMIZER="$2"; shift ;;
        --colbert-url)      COLBERT_URL="$2"; shift ;;
        --sglang-port)      SGLANG_PORT="$2"; shift ;;
        --train-size)       TRAIN_SIZE="$2"; shift ;;
        --dev-size)         DEV_SIZE="$2"; shift ;;
        --test-size)        TEST_SIZE="$2"; shift ;;
        --max-iters)        MAX_ITERS="$2"; shift ;;
        --encoder-device)   ENCODER_DEVICE="$2"; shift ;;
        --baseline)         BASELINE=true ;;
        --no-visuals)       NO_VISUALS=true ;;
        --sample-trajectory) SAMPLE_TRAJECTORY=true ;;
        --seed)             SEED="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model             LM model name. Default: Qwen/Qwen2.5-7B-Instruct"
            echo "  --optimizer         Prompt optimizer: clusterfs | miprov2 | bfrs. Default: clusterfs"
            echo "  --colbert-url       ColBERTv2 API endpoint. Default: http://localhost:8894/api/search"
            echo "  --sglang-port       sglang server port for local HF models (e.g. 7501)"
            echo "  --train-size        Number of training examples. Default: 500"
            echo "  --dev-size          Number of validation examples. Default: 200"
            echo "  --test-size         Number of test examples. Default: 500"
            echo "  --max-iters         Max ReAct steps per question. Default: 20"
            echo "  --encoder-device    SentenceTransformer device (cpu / cuda). Default: cpu"
            echo "  --baseline          Evaluate zero-shot agent only (skip optimization)"
            echo "  --no-visuals        Disable matplotlib cluster plots"
            echo "  --sample-trajectory Print a qualitative trajectory comparison at the end"
            echo "  --seed              Random seed for reproducibility"
            echo ""
            echo "Supported models:"
            for m in "${VALID_MODELS[@]}"; do echo "  $m"; done
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate model
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Invalid model: $MODEL"
    echo "Supported models:"
    for m in "${VALID_MODELS[@]}"; do echo "  $m"; done
    exit 1
fi

# Set AUTO_CONFIRM for non-interactive runs
export PYTHONUNBUFFERED=1
export AUTO_CONFIRM=true

# Build log file name
MODEL_ID="_$(basename "$MODEL")"
if [[ "$BASELINE" == "true" ]]; then
    RUN_TAG="_baseline"
else
    RUN_TAG="_${OPTIMIZER}"
fi
SEED_TAG=""
[[ -n "$SEED" ]] && SEED_TAG="_s${SEED}"
LOG_FILE="react_agent_experiment${MODEL_ID}${RUN_TAG}${SEED_TAG}_$(date +'%Y-%m-%d').log"

# Build command arguments
CMD_ARGS=(
    --model "$MODEL"
    --optimizer "$OPTIMIZER"
    --colbert-url "$COLBERT_URL"
    --train-size "$TRAIN_SIZE"
    --dev-size "$DEV_SIZE"
    --test-size "$TEST_SIZE"
    --max-iters "$MAX_ITERS"
    --encoder-device "$ENCODER_DEVICE"
)

[[ -n "$SGLANG_PORT" ]]      && CMD_ARGS+=(--sglang-port "$SGLANG_PORT")
[[ "$BASELINE" == "true" ]]  && CMD_ARGS+=(--baseline)
[[ "$NO_VISUALS" == "true" ]] && CMD_ARGS+=(--no-visuals)
[[ "$SAMPLE_TRAJECTORY" == "true" ]] && CMD_ARGS+=(--sample-trajectory)
[[ -n "$SEED" ]] && CMD_ARGS+=(--seed "$SEED")

# PATCH: raise open-file limit to avoid LiteLLM sqlite issues
ulimit -n 65535

# Run experiment
nohup python3.11 react_agent_experiment.py "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE" &

echo ""
echo "ReAct Agent Experiment"
echo "----------------------"
echo "Model         : $MODEL"
echo "Optimizer     : $OPTIMIZER"
echo "ColBERT URL   : $COLBERT_URL"
echo "Train / Dev / Test : $TRAIN_SIZE / $DEV_SIZE / $TEST_SIZE"
echo "Max iters     : $MAX_ITERS"
echo "Encoder device: $ENCODER_DEVICE"
echo "Seed          : ${SEED:-<auto>}"
echo "Baseline only : $BASELINE"
echo "Log file      : $LOG_FILE"
echo ""
echo "To monitor:  tail -f $LOG_FILE"
echo ""
