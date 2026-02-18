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
# 2. Remove old uv tool installation (if exists)
#############################################

if uv tool list 2>/dev/null | grep -q colbert-server; then
    echo "[INFO] Removing old uv tool colbert-server installation..."
    uv tool uninstall colbert-server
fi

#############################################
# 3. Install Python 3.13 via uv (if needed)
#############################################

echo "[INFO] Ensuring Python 3.13 is available via uv..."
uv python install 3.13

#############################################
# 4. Create venv with Python 3.13
#############################################

# Remove old venv if it exists (to ensure fresh install with new dependencies)
if [ -d "$VENV_NAME" ]; then
    echo "[INFO] Removing old virtualenv to ensure clean install..."
    rm -rf "$VENV_NAME"
fi

echo "[INFO] Creating virtualenv '$VENV_NAME' with Python 3.13..."
uv venv "$VENV_NAME" --python 3.13

echo "[INFO] Activating virtualenv..."
source "$VENV_NAME/bin/activate"

#############################################
# 5. Install colbert-server with compatible dependencies
#############################################

echo "[INFO] Installing colbert-server with compatible transformers version..."
# Pin transformers to 4.49.0 - colbert-server is NOT compatible with transformers 5.x
# See: https://github.com/stanford-futuredata/ColBERT/issues/391
uv pip install "transformers==4.49.0"
uv pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install colbert-server

# Verify installation
colbert-server --version || echo "[INFO] colbert-server installed successfully"

#############################################
# 6. Prepare assets folder
#############################################

mkdir -p "$ASSETS_DIR"

#############################################
# 7. Run colbert-server in nohup sticky mode
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
