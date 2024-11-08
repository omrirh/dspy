#!/bin/bash

# make sure Nvidia driver is present on machine
command -v nvidia-smi >/dev/null 2>&1 || { echo >&2 "ERROR: NVIDIA drivers are missing."; exit 1; }

# Get session variables
source vm_vars.env

# login with huggingface-cli
huggingface-cli login --token $HF_TOKEN

# Spin up the local llama model
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path meta-llama/Meta-Llama-3-8B-Instruct   --port 7501