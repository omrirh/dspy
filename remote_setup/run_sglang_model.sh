#!/bin/bash

# Default model name
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Enable session variables and env
source vm_vars.env
source dspy_venv/bin/activate

# Setup SGLang & flashinfer-python
uv pip install flashinfer-python==0.2.5 torch==2.6.0+cu124 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/ --no-deps
uv pip install "sglang==0.4.6.post4"

# Make sure Nvidia driver is present on machine
command -v nvidia-smi >/dev/null 2>&1 || { echo >&2 "ERROR: NVIDIA drivers are missing."; exit 1; }

# Login with huggingface-cli
huggingface-cli login --token "$HF_TOKEN"

SERVER_CMD="python -m sglang.launch_server \
  --model-path \"$MODEL_NAME\" \
  --port 7501"

# Conditionally apply model-specific tweaks
if [[ "$MODEL_NAME" == "Qwen/Qwen3-8B" ]]; then
  echo -e "Using a dedicated reasoning parser for $MODEL_NAME model"
  SERVER_CMD+=" --reasoning-parser qwen3"
elif [[ "$MODEL_NAME" == "google/gemma-3-4b-it" ]]; then
  echo -e "Applying memory-friendly settings for $MODEL_NAME"
  SERVER_CMD+=" --context-length 8192"
fi

# Spin up the local sglang model persistently
nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  HF_TOKEN="$HF_TOKEN" \
  CUDA_HOME=/usr/local/cuda-12.4 \
  PATH=$CUDA_HOME/bin:$PATH \
  LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
  bash -c "$SERVER_CMD" | tee "sglang_run.log" &
