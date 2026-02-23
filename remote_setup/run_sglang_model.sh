#!/bin/bash
# Launch a local SGLang model server.
#
# Usage:
#   bash remote_setup/run_sglang_model.sh --model-name meta-llama/Llama-3.2-3B-Instruct

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
PORT=30000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

source vm_vars.env
source dspy_venv/bin/activate

# Install SGLang and FlashInfer if not present
uv pip install flashinfer-python==0.2.5 torch==2.6.0+cu124 \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/ --no-deps
uv pip install "sglang==0.4.6.post4"

# Require NVIDIA drivers
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: NVIDIA drivers missing."; exit 1; }

# Authenticate with HuggingFace
huggingface-cli login --token "$HF_TOKEN"

SERVER_CMD="python -m sglang.launch_server --model-path \"$MODEL_NAME\" --port $PORT"

if [[ "$MODEL_NAME" == "Qwen/Qwen3-8B" ]]; then
    SERVER_CMD+=" --reasoning-parser qwen3"
elif [[ "$MODEL_NAME" == "google/gemma-3-4b-it" ]]; then
    SERVER_CMD+=" --context-length 8192"
fi

nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    HF_TOKEN="$HF_TOKEN" \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/usr/local/cuda-12.4/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
    bash -c "$SERVER_CMD" | tee "sglang_run.log" &

echo "SGLang server launching: $MODEL_NAME on port $PORT"
echo "Tail logs with: tail -f sglang_run.log"
