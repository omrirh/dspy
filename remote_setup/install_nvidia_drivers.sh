#!/bin/bash
set -e

# NVIDIA driver 570 is the latest production branch compatible with CUDA 12.4–12.8
# and validated on Ubuntu 22.04 + A100 (SM80).
# 580.x is the upstream userspace version shipped by apt on this branch.
DRIVER_VERSION="570"
CUDA_VERSION="12.4"
CUDA_DEB="cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb"

echo "Updating system and installing required packages..."
sudo apt update
sudo apt install -y dkms build-essential linux-headers-$(uname -r) wget gnupg

echo "Installing NVIDIA driver ${DRIVER_VERSION}..."
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install -y nvidia-driver-${DRIVER_VERSION}

# Hold the driver package so apt upgrade cannot silently bump the kernel module
# version while leaving the userspace NVML library behind (causes NVML mismatch).
sudo apt-mark hold nvidia-driver-${DRIVER_VERSION}
echo "nvidia-driver-${DRIVER_VERSION} held at current version (prevents NVML mismatch on apt upgrade)"

echo "Downloading CUDA ${CUDA_VERSION} local installer..."
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/${CUDA_DEB}

echo "Installing CUDA repository package..."
sudo dpkg -i ${CUDA_DEB}

echo "Installing the CUDA GPG key..."
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

echo "Installing CUDA Toolkit ${CUDA_VERSION}..."
sudo apt update
sudo apt install -y cuda-toolkit-12-4

echo "Adding CUDA to environment variables..."
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "Done. A reboot is required to load the new kernel module."
echo "Run: sudo reboot"
echo "After reboot, verify with: nvidia-smi && nvcc --version"
