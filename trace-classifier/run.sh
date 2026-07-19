#!/usr/bin/env bash
# Evaluate the classifier, then generate figures.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
"$PY" evaluate.py
"$PY" figures.py
