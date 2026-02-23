#!/bin/bash
# Prepare a Python 3.11 virtualenv with all project dependencies.
# Run once after provisioning a new remote instance.

set -e

sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv dspy_venv
source dspy_venv/bin/activate
pip install --upgrade pip
pip install uv

# Core torch (CUDA 12.4)
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Project dependencies
uv pip install -r remote_setup/requirements.txt

echo "Virtualenv ready. Activate with: source dspy_venv/bin/activate"
