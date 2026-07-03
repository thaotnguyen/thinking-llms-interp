#!/usr/bin/env bash
# Run gpt-oss activation collection in the dedicated NATIVE-MXFP4 env (.venv-mxfp4).
# See requirements-mxfp4.txt for why this is separate from the main .venv.
#
# Usage:
#   ./run_activations_mxfp4.sh --model openai/gpt-oss-20b --layers 5 8 11 14 17 20 \
#       --n_examples 100000 --batch_size 1 --disable_cache
# Any args are passed through to train-saes/generate_activations.py.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$REPO_DIR/.venv-mxfp4/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: $VENV_PY not found. Build it first:" >&2
  echo "  uv venv --python 3.12 .venv-mxfp4" >&2
  echo "  uv pip install --python .venv-mxfp4/bin/python -r requirements-mxfp4.txt" >&2
  exit 1
fi

# HF token (needed on first load so 'kernels' can fetch the MXFP4 triton kernel from the Hub)
if [[ -f "$REPO_DIR/.env" ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$REPO_DIR/.env" | head -1 | cut -d= -f2- || true)"
  export HF_TOKEN
fi

cd "$REPO_DIR/train-saes"
exec "$VENV_PY" generate_activations.py "$@"
