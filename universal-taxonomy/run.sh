#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${OPENAI_API_KEY:?set OPENAI_API_KEY in the environment or .env}"
N_TRIALS="${N_TRIALS:-1500}"
WORKERS="${WORKERS:-4}"
PY="${PYTHON:-python}"
export N_TRIALS

# Stage 1 encodes the SAEs from the matthewshu/medcasereasoning-cot-activations snapshot
# (override with $SAE_SNAPSHOT); it is resumable and writes to results/vars/latents.
echo "== [1/5] encode SAE latents (L) =="
"$PY" encode_latents.py --workers "$WORKERS"
echo "== [2/5] embed sentences (S) =="
"$PY" embed_sentences.py
echo "== [3/5] cache latents (E=LᵀS) =="
"$PY" cache_latents.py
echo "== [4/5] Optuna search (k*coverage, N_TRIALS=$N_TRIALS) =="
"$PY" search.py
echo "== [5/5] title the taxonomy =="
"$PY" title.py

echo "== done -> results/plain/taxonomy.json =="
