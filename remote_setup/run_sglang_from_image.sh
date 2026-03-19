#!/bin/bash
# Deploy SGLang server from a Docker image.
# Use this for models too new for the pinned pip stack (e.g. Qwen3.5-9B).
# Prerequisites: run remote_setup/install_docker.sh once first.
#
# Usage:
#   bash remote_setup/run_sglang_from_image.sh --model-name Qwen/Qwen3.5-9B
#   bash remote_setup/run_sglang_from_image.sh --model-name Qwen/Qwen3.5-9B --image lmsysorg/sglang:dev

# Default to today's nightly (CUDA 12.x, SM80-compatible).
# Override with --image for a specific tag or 'lmsysorg/sglang:dev' for always-latest.
MODEL_NAME="Qwen/Qwen3.5-9B"
PORT=30000
IMAGE="lmsysorg/sglang:nightly-dev-20260304-c18cff4f"
CONTAINER_NAME="sglang_server"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --image)      IMAGE="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

source vm_vars.env

# Require Docker
command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker not found. Run: bash remote_setup/install_docker.sh"; exit 1; }

# Require NVIDIA drivers
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: NVIDIA drivers missing."; exit 1; }

# Reuse local HF cache to avoid re-downloading weights
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# Stop any previously running container
docker rm -f "$CONTAINER_NAME" 2>/dev/null

# Model-specific flags
MODEL_FLAGS="--model-path $MODEL_NAME --port $PORT --host 0.0.0.0"
if [[ "$MODEL_NAME" == Qwen/Qwen3* ]]; then
    MODEL_FLAGS+=" --reasoning-parser qwen3"
elif [[ "$MODEL_NAME" == "google/gemma-3-4b-it" ]]; then
    MODEL_FLAGS+=" --context-length 8192"
fi

nohup docker run --rm \
    --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    -p "$PORT:$PORT" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HF_TOKEN="$HF_TOKEN" \
    "$IMAGE" \
    python -m sglang.launch_server $MODEL_FLAGS \
    > sglang_docker.log 2>&1 &

echo "SGLang Docker server launching: $MODEL_NAME on port $PORT"
echo "Image: $IMAGE"
echo "Tail logs with: tail -f sglang_docker.log"
