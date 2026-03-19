#!/bin/bash
# Install Docker CE + NVIDIA Container Toolkit on Ubuntu.
# Run once (as root) before using run_sglang_from_image.sh.

set -e

# --- Docker CE ---
if command -v docker &>/dev/null; then
    echo "Docker already installed: $(docker --version)"
else
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io
    echo "Docker installed: $(docker --version)"
fi

# --- NVIDIA Container Toolkit ---
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "NVIDIA Container Toolkit already installed."
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "NVIDIA Container Toolkit installed."
fi

echo ""
echo "Setup complete. Run:"
echo "  bash remote_setup/run_sglang_from_image.sh --model-name Qwen/Qwen3.5-9B"
