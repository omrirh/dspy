#!/bin/bash

set -e

# Define the mount point
MOUNT_DIR=$1

# Define environment variables for cache directories and virtual environment
export TRANSFORMERS_CACHE="$MOUNT_DIR/.cache/transformers"
export HF_DATASETS_CACHE="$MOUNT_DIR/.cache/datasets"
VENV_DIR="$MOUNT_DIR/venv"

# Create directories if they don't exist
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VENV_DIR"

# Inform the user of cache locations
echo "Setting transformers cache directory to: $TRANSFORMERS_CACHE"
echo "Setting datasets cache directory to: $HF_DATASETS_CACHE"

# Install virtual environment in specified directory
echo "Creating virtual environment in: $VENV_DIR"
python3 -m venv "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Confirm the setup
echo "Environment setup complete."
echo "Transformers cache: $TRANSFORMERS_CACHE"
echo "Datasets cache: $HF_DATASETS_CACHE"
echo "Virtual environment located at: $VENV_DIR"
