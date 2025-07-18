#!/bin/bash

# Update package list and install prerequisites
sudo apt update && sudo apt install -y software-properties-common

# Add deadsnakes PPA for newer Python versions
sudo add-apt-repository -y ppa:deadsnakes/ppa

# Update package list again
sudo apt update

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Create virtualenv for project dependencies
python3.11 -m venv dspy_venv
source dspy_venv/bin/activate
pip install --upgrade pip
pip install uv
uv pip install --upgrade pip

# Install torch packages from cuda 12.4 wheel independently (core dependency)
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124  --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install -r dspy/remote_setup/requirements.txt
