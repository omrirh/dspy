#!/bin/bash

source dspy_venv/bin/activate
source vm_vars.env

nohup env CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path /results \
  --port 7501 > trained_llama_run.log 2>&1 &