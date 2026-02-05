#!/usr/bin/env bash
set -euo pipefail

#############################################
# CONFIG
#############################################

PORT=8894
ASSETS_DIR="/tmp/wiki-assets"
VENV_NAME="colbert_venv"
LOGFILE="colbert_server.log"

#############################################
# 1. Install uv (only if missing)
#############################################

if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[INFO] uv already installed."
fi

#############################################
# 2. Create venv + install pip + jinja2 (req for runtime)
#############################################

if [ ! -d "$VENV_NAME" ]; then
    echo "[INFO] Creating virtualenv '$VENV_NAME'..."
    python3.11 -m venv "$VENV_NAME"
fi

echo "[INFO] Activating virtualenv..."
source "$VENV_NAME/bin/activate"

echo "[INFO] Upgrading pip + installing jinja2..."
pip install --upgrade pip
pip install jinja2

#############################################
# 3. Install colbert-server using uv
#############################################

echo "[INFO] Installing colbert-server via uv..."
uv tool install colbert-server

# Ensure uv-installed binaries are on PATH for this shell
export PATH="$HOME/.local/share/uv/tools/colbert-server/bin:$PATH"

# Ensure colbert-server is ready for running
colbert-server doctor

#############################################
# 4. Prepare assets folder
#############################################

mkdir -p "$ASSETS_DIR"

#############################################
# 5. Run colbert-server in nohup sticky mode
#############################################
echo "[INFO] Starting colbert-server"
echo "[INFO] Logs will be written to: $LOGFILE"

# Kill any existing colbert-server on this port
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[WARN] Port $PORT already in use. Killing old process..."
    kill -9 "$(lsof -t -i:$PORT)"
fi

CUDA_VISIBLE_DEVICES="" nohup colbert-server serve \
    --download-archives "$ASSETS_DIR" \
    --extract \
    --port "$PORT" \
    > "$LOGFILE" 2>&1 &

echo "[INFO] colbert-server started."
tail -f "$LOGFILE"
