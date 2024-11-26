#!/bin/bash

# Enable session variables and env
source vm_vars.env
source dspy_venv/bin/activate

# Setup SGLang
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
pip install huggingface_hub

# make sure Nvidia driver is present on machine
command -v nvidia-smi >/dev/null 2>&1 || { echo >&2 "ERROR: NVIDIA drivers are missing."; exit 1; }

# login with huggingface-cli
huggingface-cli login --token $HF_TOKEN

# Spin up the local llama model persistently
nohup env CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 7501 > llama_run.log 2>&1 &
