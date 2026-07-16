#!/usr/bin/env bash
# RUN THIS ON YOUR LOCAL COMPUTER (not on the vast.ai box).
#
# It PULLS the new work products from the rented GPU box into your local copy at
# /home/ttn/Development/med-interp/thinking-llms-interp/. rsync is incremental, so
# it is safe to run repeatedly -- re-run it after the titles finish and it will
# only copy what changed.
#
# Why "pull" and not "push": the box cannot reach your laptop (your laptop is the
# SSH client, behind NAT). So the transfer has to be initiated from your side.
set -euo pipefail

# --- your SSH connection to the box (same as your interactive ssh) ---
SSH='ssh -i ~/.ssh/vastai -p 31827'
REMOTE='root@ssh6.vast.ai:/workspace/Documents/med-interp/thinking-llms-interp/'
LOCAL="$HOME/Development/med-interp/thinking-llms-interp/"

# Excludes: the huge, re-downloadable-from-HF inputs and local-only build dirs.
# (The activation pkls are 73 GB and come from matthewshu/medcasereasoning-cot-
#  activations; the GGUF/venvs are box-local. Add them below if you want them --
#  see the second command at the bottom.)
rsync -avz --progress --partial --human-readable \
  -e "$SSH" \
  --exclude 'activations_*.pkl' \
  --exclude '*.gguf' \
  --exclude '.venv*' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '*.tmp*' \
  "$REMOTE" "$LOCAL"

echo
echo "Done. Pulled the work products (~43 GB): annotate-saes/ (scripts, titles,"
echo "sidecars, dedup SAEs, WORKLOG) + generate-responses/.../annotated_responses_*.json"
echo "(both the matthewshu/HF-SAE set and the dedup set) + modified utils/."
echo
echo "If you ALSO want the 73 GB activation pkls (the exact space the SAEs consume),"
echo "run this once, separately:"
echo
echo "  rsync -avz --progress --partial -e \"$SSH\" \\"
echo "    --include='*/' --include='activations_*.pkl' --exclude='*' \\"
echo "    '$REMOTE' '$LOCAL'"
