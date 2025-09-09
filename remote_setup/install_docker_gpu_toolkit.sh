#!/bin/bash

apt install -y docker.io

# repo & key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
 | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
 | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
 | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# configure Docker runtime + restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
