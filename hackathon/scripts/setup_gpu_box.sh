#!/usr/bin/env bash
# One-shot setup on a fresh Linux GPU machine (SSH box, cloud VM, or a Jupyter terminal).
#   curl/clone the repo, then:  bash hackathon/scripts/setup_gpu_box.sh
set -euo pipefail

REPO_URL="https://github.com/JoeyBryson/pythtb_gpu_accelaration_project.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$HERE/pyproject.toml" ]]; then
  echo "Run this from inside a clone of $REPO_URL" >&2
  exit 1
fi

nvidia-smi || echo "WARNING: nvidia-smi not found, no GPU visible on this machine"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$HERE"
uv sync                      # Python 3.12+ venv in hackathon/.venv, PythTB installed editable
uv run python scripts/check_env.py
uv run pytest -q
