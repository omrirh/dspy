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
elif [[ "$MODEL_NAME" == "Qwen/Qwen3-32B" ]]; then
  echo -e "Using reasoning parser + A100 80GB memory settings for $MODEL_NAME"
  SERVER_CMD+=" --reasoning-parser qwen3 --dtype bfloat16 --mem-fraction-static 0.88 --context-length 4096 --enable-torch-compile"
elif [[ "$MODEL_NAME" == "Qwen/Qwen2.5-14B-Instruct" ]]; then
  # 14B BF16 weights ~28 GB on A100 80 GB.
  # --mem-fraction-static 0.80 : 64 GB for weights + CUDA graph buffers,
  #   ~16 GB remaining for KV cache.
  echo -e "Applying A100 80GB settings for $MODEL_NAME"
  SERVER_CMD+=" --dtype bfloat16 --mem-fraction-static 0.80 --context-length 16384 --enable-torch-compile"
elif [[ "$MODEL_NAME" == "Qwen/Qwen2.5-32B-Instruct" ]]; then
  # 32B BF16 weights ~64 GB on A100 80 GB.
  # --mem-fraction-static 0.88 : 70.4 GB for weights + CUDA graph buffers,
  #   ~9.6 GB remaining for KV cache.
  # --context-length 8192      : ClusterFewshot-selected demos can exceed 4K;
  #   8K headroom avoids 400 Bad Request errors during optimizer eval.
  # --enable-torch-compile     : ~15 % throughput gain on A100, +60 s warm-up.
  echo -e "Applying A100 80GB memory settings for $MODEL_NAME"
  SERVER_CMD+=" --dtype bfloat16 --mem-fraction-static 0.88 --context-length 8192 --enable-torch-compile"
elif [[ "$MODEL_NAME" == "Qwen/Qwen2.5-32B-Instruct-AWQ" ]]; then
  # 32B AWQ 4-bit weights ~20 GB on A100 80 GB.
  # --quantization awq         : load pre-quantized AWQ weights.
  # --dtype float16            : AWQ dequantizes to fp16 for computation.
  # --mem-fraction-static 0.85 : 68 GB allocated; ~48 GB available for KV cache after ~20 GB weights.
  # --context-length 16384     : generous context headroom given large KV cache budget.
  # --enable-torch-compile     : ~15 % throughput gain on A100, +60 s warm-up.
  echo -e "Applying AWQ 4-bit settings for $MODEL_NAME"
  SERVER_CMD+=" --quantization awq --dtype float16 --mem-fraction-static 0.85 --context-length 16384 --enable-torch-compile"
elif [[ "$MODEL_NAME" == "meta-llama/Llama-3.3-70B-Instruct" ]]; then
  # 70B requires FP8 quantization to fit in 80 GB.
  echo -e "Applying FP8 quantization + A100 80GB memory settings for $MODEL_NAME"
  SERVER_CMD+=" --quantization fp8 --dtype bfloat16 --mem-fraction-static 0.93 --context-length 4096"
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
