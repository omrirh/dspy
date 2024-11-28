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

# Install poetry dependencies
cd dspy
pip install poetry
poetry env use python3.11
poetry install

# Install pip dependencies
pip install -r remote_setup/requirements.txt
