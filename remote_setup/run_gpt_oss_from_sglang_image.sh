#!/usr/bin/env bash
set -euo pipefail

# Host-side setup
source vm_vars.env
export MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"     # or lmsys/gpt-oss-20b-bf16
export CONTAINER_NAME="${CONTAINER_NAME:-sglang-oss}"
export IMAGE_TAG="${IMAGE_TAG:-lmsysorg/sglang:v0.5.0rc2-cu126}"  # GPT-OSS day-0 image (CUDA 12.6)
export HF_CACHE_HOST="${HF_CACHE_HOST:-$HOME/.cache/huggingface}" # host cache mount

# Auto-detect GPU arch for faster kernel builds (A100=8.0, L4=8.9, etc.)
GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 || true)"
[[ -z "${GPU_ARCH:-}" ]] && GPU_ARCH="8.0"   # sane default (A100)

# Pre-pull (resumes if previously interrupted)
docker pull "$IMAGE_TAG" >/dev/null

# Clean up any old container
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# Run
docker run -d --name "$CONTAINER_NAME" \
  --gpus "device=0" \
  --ipc=host --shm-size=2g --restart unless-stopped \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HF_HOME="$HF_HOME"\
  -e TRANSFORMERS_CACHE="$HF_HOME/hub" \
  -p 7501:7501 \
  -v "$HF_CACHE_HOST:$HF_CACHE_HOST" \
  "$IMAGE_TAG" \
  bash -lc '
    set -euo pipefail
    : "${MODEL_NAME:=openai/gpt-oss-20b}"

    # Build args as an array to avoid quote issues
    args=(python -m sglang.launch_server --model-path "$MODEL_NAME" --host 0.0.0.0 --port 7501)

    # Model-specific tweaks (mirror your bare-metal script)
    if [[ "$MODEL_NAME" == "Qwen/Qwen3-8B" ]]; then
      args+=(--reasoning-parser qwen3)
    elif [[ "$MODEL_NAME" == "google/gemma-3-4b-it" ]]; then
      args+=(--context-length 8192)
    fi

    # For BF16 checkpoint, trim KV-cache VRAM (usually faster/better on A100-40GB)
    if [[ "$MODEL_NAME" == "lmsys/gpt-oss-20b-bf16" ]]; then
      args+=(--kv-cache-dtype fp8_e5m2)
    fi

    echo "Launching: ${args[*]}"
    exec "${args[@]}"
  '

echo "Container started: $CONTAINER_NAME"
echo "View logs: docker logs -f $CONTAINER_NAME"
