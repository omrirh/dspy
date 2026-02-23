#!/bin/bash
# Install NVIDIA driver 550 and CUDA 12.4 on Ubuntu 22.04.
# Run once on a fresh GPU instance before prepare_virtualenv.sh.

set -e

echo "Updating system and installing prerequisites ..."
sudo apt update
sudo apt install -y dkms build-essential linux-headers-$(uname -r) wget gnupg

echo "Adding NVIDIA driver PPA ..."
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install -y nvidia-driver-550

echo "Downloading CUDA 12.4.1 installer ..."
CUDA_DEB=cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/$CUDA_DEB

echo "Installing CUDA repository package ..."
sudo dpkg -i $CUDA_DEB
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

echo "Installing CUDA Toolkit 12.4 ..."
sudo apt update
sudo apt install -y cuda-toolkit-12-4

echo "Setting environment variables ..."
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "Done. Run: source ~/.bashrc && nvcc --version"
