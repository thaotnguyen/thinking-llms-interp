# trace-classifier

Predicts whether a chain-of-thought reasoning trace led to a **correct** answer, using
its **universal-taxonomy sentence sequence** (categories 0..4) as the only signal, for
the 6 recommended-decoding models. Reports lift over the always-guess-correct baseline.

## Layout
- `features.py`  — feature generation: loads taxonomy sequences + labels, builds 81
  length-normalized features per trace (`features.load()`).
- `evaluate.py`  — XGBoost under 3 protocols, each with per-group lift → `results/metrics.json`:
  10-fold grouped CV (by pmcid), leave-one-dataset-out, leave-one-model-out.
- `figures.py`   — `results/figures/{lift,importance}.png` from the metrics.
- `run.sh`       — runs evaluate then figures.

## Run
```bash
./run.sh                  # or: python evaluate.py && python figures.py
```

## Prerequisites
1. `universal-taxonomy/run.sh` completed → `results_recommended/plain/{labels,latent_cache}`.
2. Rec responses graded by the repo's `grade_responses.py` →
   `trace-classifier/data/graded/responses_{model}.rec_graded.json` (provides `is_correct`).
