#!/usr/bin/env python3
"""Evaluate XGBoost on the taxonomy features under three protocols, reporting lift
over the always-guess-correct baseline for every group:

  1. cv       - grouped 10-fold CV (folds split by pmcid); lift overall and per model
  2. dataset  - leave-one-dataset-out (train on 2 question sources, test on the 3rd)
  3. model    - leave-one-model-out (train on 5 models, test on the held-out model)

Baseline = predict "correct" for everything: its accuracy is the positive rate and
its AUC is 0.5. Writes results/metrics.json. Run: `python evaluate.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import features as F

RES = Path(__file__).resolve().parent / "results"


def xgb(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=5,
        scale_pos_weight=scale_pos_weight, eval_metric="auc", n_jobs=-1, random_state=42)


def _spw(y: np.ndarray) -> float:
    return (y == 0).sum() / max((y == 1).sum(), 1)


def _best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximizing accuracy on the training split."""
    grid = np.linspace(0.05, 0.95, 181)
    return float(grid[np.argmax([((s >= t).astype(int) == y).mean() for t in grid])])


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    pos = float(y.mean())
    base = max(pos, 1 - pos)                     # always-guess-same (majority class) accuracy
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")
    acc = float((pred == y).mean())
    return {"n": int(len(y)), "pos_rate": pos, "baseline_acc": base, "auc": auc,
            "auc_lift": auc - 0.5, "acc": acc, "acc_lift": acc - base}


def cv(d: F.Data, n_splits: int = 10) -> dict:
    """Grouped 10-fold CV by pmcid; overall out-of-fold accuracy/AUC and lift."""
    score = np.zeros(len(d.y))
    pred = np.zeros(len(d.y), int)
    for tr, va in GroupKFold(n_splits=n_splits).split(d.X, d.y, groups=d.pmcid):
        m = xgb(_spw(d.y[tr])).fit(d.X[tr], d.y[tr])
        thr = _best_threshold(d.y[tr], m.predict_proba(d.X[tr])[:, 1])
        score[va] = m.predict_proba(d.X[va])[:, 1]
        pred[va] = (score[va] >= thr).astype(int)
    return metrics(d.y, score, pred)


def leave_one_out(d: F.Data, key: np.ndarray) -> dict:
    """Train on all groups but one, test on the held-out group; lift per held-out group."""
    out = {}
    for g in sorted(np.unique(key)):
        tr, te = key != g, key == g
        m = xgb(_spw(d.y[tr])).fit(d.X[tr], d.y[tr])
        s_tr = m.predict_proba(d.X[tr])[:, 1]
        thr = _best_threshold(d.y[tr], s_tr)
        s_te = m.predict_proba(d.X[te])[:, 1]
        out[str(g)] = metrics(d.y[te], s_te, (s_te >= thr).astype(int))
    return out


def _print(title: str, rows: dict) -> None:
    print(f"\n== {title} ==")
    print(f"{'group':20s} {'n':>6s} {'same':>6s} {'AUC':>6s} {'AUClift':>8s} {'acc':>6s} {'accLift':>8s}")
    for g, m in rows.items():
        print(f"{g:20s} {m['n']:6d} {m['baseline_acc']:6.3f} {m['auc']:6.3f} "
              f"{m['auc_lift']:+8.3f} {m['acc']:6.3f} {m['acc_lift']:+8.3f}")


def main() -> None:
    RES.mkdir(exist_ok=True)
    d = F.load()
    print(f"{d.X.shape[0]} traces x {d.X.shape[1]} features | correct rate {d.y.mean():.3f}")
    results = {
        "cv": cv(d),
        "dataset": leave_one_out(d, d.source),
        "model": leave_one_out(d, d.model),
    }
    _print("10-fold CV", {"overall": results["cv"]})
    _print("leave-one-dataset-out", results["dataset"])
    _print("leave-one-model-out", results["model"])
    json.dump(results, open(RES / "metrics.json", "w"), indent=2)
    print(f"\nwrote {RES / 'metrics.json'}")


if __name__ == "__main__":
    main()
