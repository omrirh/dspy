#!/bin/bash

# Check for model name argument
if [ -z "$1" ]; then
  echo "Usage: $0 <model-name>"
  echo "Example: $0 meta-llama/Meta-Llama-3-8B-Instruct"
  exit 1
fi

MODEL_NAME=$1

# Enable session variables and env
source vm_vars.env
source dspy_venv/bin/activate

# Setup SGLang & flashinfer-python (updated requirements.txt takes care of all)
pip install flashinfer-python -f https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
pip install "sglang==0.4.3"

# make sure Nvidia driver is present on machine
command -v nvidia-smi >/dev/null 2>&1 || { echo >&2 "ERROR: NVIDIA drivers are missing."; exit 1; }

# login with huggingface-cli
huggingface-cli login --token $HF_TOKEN

# Spin up the local llama model persistently
nohup env CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL_NAME" \
  --port 7501  | tee "${MODEL_NAME//\//_}_run.log" &
