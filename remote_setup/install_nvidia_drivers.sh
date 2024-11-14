#!/bin/bash

set -e

echo "Updating system and installing required packages..."
sudo apt update
sudo apt install -y dkms build-essential linux-headers-$(uname -r)

echo "Ensuring non-free repository is enabled..."
# Add contrib and non-free components to sources.list if not already present
if ! grep -q "non-free" /etc/apt/sources.list; then
  sudo sed -i '/^deb/s/$/ contrib non-free non-free-firmware/' /etc/apt/sources.list
  echo "Non-free repository added. Updating package list..."
  sudo apt update
else
  echo "Non-free repository already enabled."
fi

# enabling graphics driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

echo "Installing NVIDIA driver..."
sudo apt install -y nvidia-driver-535

echo "Finished installing NVIDIA driver."
