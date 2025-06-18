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

# Setup SGLang & flashinfer-python (updated requirements.txt takes care of all)
pip install flashinfer-python==0.2.5 torch==2.5.1 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.5/ --no-deps
pip install "sglang==0.4.3"

# Make sure Nvidia driver is present on machine
command -v nvidia-smi >/dev/null 2>&1 || { echo >&2 "ERROR: NVIDIA drivers are missing."; exit 1; }

# Login with huggingface-cli
huggingface-cli login --token "$HF_TOKEN"

# Spin up the local sglang model persistently
nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  HF_TOKEN="$HF_TOKEN" \
  python -m sglang.launch_server \
  --model-path "$MODEL_NAME" \
  --port 7501 | tee "sglang_run.log" &
