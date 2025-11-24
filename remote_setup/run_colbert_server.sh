#!/usr/bin/env bash
set -euo pipefail

#-----------------------------#
# CONFIG                     #
#-----------------------------#
ASSETS_DIR="/tmp/wiki-assets"
SERVER_PORT=8894
LOGFILE="/var/log/colbert-server.log"

#-----------------------------#
# Install uv if missing       #
#-----------------------------#
if ! command -v uv &>/dev/null; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "[INFO] uv already installed."
fi

export PATH="$HOME/.local/bin:$PATH"

#-----------------------------#
# Python 3.11 environment     #
#-----------------------------#
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python 3.11 virtual environment..."
    python3.11 -m venv venv
fi

echo "[INFO] Activating venv..."
source venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

echo "[INFO] Installing runtime dependencies..."
pip install jinja2

#-----------------------------#
# Install colbert-server      #
#-----------------------------#
echo "[INFO] Installing colbert-server via uv..."
uv tool install colbert-server

#-----------------------------#
# Prepare assets directory    #
#-----------------------------#
mkdir -p "$ASSETS_DIR"

#-----------------------------#
# Start the server in sticky mode
#-----------------------------#
echo "[INFO] Starting ColBERTv2 server on GPU 1 (nohup)..."
echo "[INFO] Logs: $LOGFILE"

nohup bash -c "
    source venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 uvx colbert-server serve \
        --download-archives '$ASSETS_DIR' \
        --extract \
        --port '$SERVER_PORT'
" >> "$LOGFILE" 2>&1 &

echo "[INFO] ColBERTv2 server started in background."
echo "[INFO] PID: $!"
