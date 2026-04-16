#!/bin/bash
# Prepare a Python 3.11 virtualenv with all project dependencies.
# Run once after provisioning a new remote instance.
#
# Validated stack: sglang==0.4.6.post4, CUDA 12.4, SM80 (Ampere), torch 2.6.0+cu124
#
# Creates the venv at <repo-parent>/dspy_venv so it lives outside the repo
# and is never tracked by git.
#
# Usage (always run from the parent of the repo):
#   bash dspy/remote_setup/prepare_virtualenv.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$(realpath -m "$REPO_ROOT/../dspy_venv")"

if ! command -v python3.11 &>/dev/null || ! python3.11 -c "import venv" &>/dev/null 2>&1; then
    sudo apt update && sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

echo "Creating venv at: $VENV_DIR"
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install uv

# Core torch + torchvision — must come from the PyTorch CUDA index.
# torchvision must match torch exactly; the PyPI CPU build breaks torchvision::nms.
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# FlashInfer — must use its own index with --no-deps to avoid pulling in a
# mismatched torch from PyPI.
uv pip install flashinfer-python==0.2.5 \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/ --no-deps

# SGLang — install base package only (runtime deps are pinned in requirements.txt)
uv pip install "sglang==0.4.6.post4"

# All remaining dependencies (pinned versions, see requirements.txt for rationale).
# requirements.txt contains `-e .` which installs DSPy from the repo in editable
# mode — pip resolves `.` relative to CWD, so we set CWD to the repo root.
(cd "$REPO_ROOT" && uv pip install -r "$REPO_ROOT/remote_setup/requirements.txt")

echo ""
echo "Virtualenv ready at: $VENV_DIR"
echo "Activate with:       source $VENV_DIR/bin/activate"
